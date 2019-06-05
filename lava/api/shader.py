# -*- coding: UTF-8 -*-

import lava.api.vulkan as vk
from lava.api.bytecode import ByteCode, ByteCodeError
from lava.api.bytes import Array, Matrix, Scalar, Struct, Vector
from lava.api.constants.spirv import Access, Decoration, ExecutionMode, ExecutionModel, Layout, Order, StorageClass
from lava.api.constants.vk import BufferUsage
from lava.api.util import Destroyable, LavaError, LavaUnsupportedError


class Shader(Destroyable):

    def __init__(self, device, path, entry_point=None):
        super(Shader, self).__init__()
        self.device = device
        with open(path, "rb") as f:
            self.bytez = f.read()
            self.handle = vk.vkCreateShaderModule(
                self.device.handle, vk.VkShaderModuleCreateInfo(codeSize=len(self.bytez), pCode=self.bytez), None)

        self.byte_code = ByteCode(self.bytez)

        # placeholder for inspection variables
        self.definitions_block = None
        self.block_data = None

        # check / set entry point
        self.entry_point, self.entry_point_index = self.check_entry_point(entry_point)
        self.local_size = self.check_local_size(self.entry_point_index)

    def _destroy(self):
        vk.vkDestroyShaderModule(self.device.handle, self.handle, None)

    def check_entry_point(self, entry_point):
        entry_points_detected = self.byte_code.find_entry_points(execution_model=ExecutionModel.GL_COMPUTE)
        index = None

        if len(entry_points_detected) == 0:
            raise RuntimeError("Could not find entry points for execution model {}".format(ExecutionModel.GL_COMPUTE))

        if entry_point is not None and entry_point not in entry_points_detected.values():
            raise RuntimeError("Could not find entry point {} in detected entry points {}".format(
                entry_point, ", ".join(entry_points_detected)))

        if entry_point is None:
            if len(entry_points_detected) > 1:
                raise RuntimeError("Multiple entry points found {}".format(", ".join(entry_points_detected.values())))
            entry_point = list(entry_points_detected.values())[0]

        for index_candidate, entry_point_candidate in entry_points_detected.items():
            if entry_point_candidate == entry_point:
                index = index_candidate

        return entry_point, index

    def check_local_size(self, entry_point_index):
        execution_mode, literals = self.byte_code.find_entry_point_details(entry_point_index)

        if execution_mode != ExecutionMode.LOCAL_SIZE:
            raise LavaUnsupportedError("Unsupported execution mode {}".format(execution_mode))

        return literals

    def get_entry_point(self):
        return self.entry_point

    def get_local_size(self):
        return self.local_size

    def inspect(self):
        self.block_data = self.byte_code.find_blocks()
        self.definitions_block = {}

        defs_scalar = {index: Scalar.of(dtype) for index, dtype in self.byte_code.types_scalar.items()}
        defs_vector = {index: Vector(n, dtype) for index, (dtype, n) in self.byte_code.types_vector.items()}
        defs_array = {}
        defs_struct = {}

        for binding in self.get_bindings():
            index, _ = self.get_block_index(binding)
            self.deduce_definition(index, defs_scalar, defs_vector, defs_array, defs_struct, index)
            self.definitions_block[index] = defs_struct[index]
            self.deduce_layout(binding)

    def deduce_definition(self, index, definitions_scalar, definitions_vector, definitions_array, definitions_struct,
                          last_struct=None):
        default_layout = Layout.STD140

        if index in self.byte_code.types_array:
            type_index, dims = self.byte_code.types_array[index]

            # build missing definition
            if type_index in self.byte_code.types_struct and type_index not in definitions_struct:
                self.deduce_definition(type_index, definitions_scalar, definitions_vector, definitions_array,
                                       definitions_struct, last_struct)

            definition = None

            # matrix types are shared, but still affected by the layout, create a instance for every occurrence
            if type_index in self.byte_code.types_matrix:
                definition = self.build_matrix_definition(type_index, default_layout, last_struct)

            definition = definition or definitions_scalar.get(type_index)
            definition = definition or definitions_vector.get(type_index)
            definition = definition or definitions_struct.get(type_index)

            definitions_array[index] = Array(definition.copy(), dims, default_layout)

        elif index in self.byte_code.types_struct:
            member_indices = self.byte_code.types_struct[index]

            # build missing definitions
            for member_index in member_indices:
                is_struct = member_index in self.byte_code.types_struct
                is_array = member_index in self.byte_code.types_array

                defs = {
                    "definitions_scalar": definitions_scalar,
                    "definitions_vector": definitions_vector,
                    "definitions_array": definitions_array,
                    "definitions_struct": definitions_struct
                }

                if is_struct and member_index not in definitions_struct:
                    self.deduce_definition(member_index, last_struct=member_index, **defs)

                if is_array and member_index not in definitions_array:
                    self.deduce_definition(member_index, last_struct=index, **defs)

            definitions = []
            for member_index in member_indices:
                definition = None

                # matrix types are shared, but still affected by the layout, create a instance for every occurrence
                if member_index in self.byte_code.types_matrix:
                    definition = self.build_matrix_definition(member_index, default_layout, last_struct)

                definition = definition or definitions_scalar.get(member_index)
                definition = definition or definitions_vector.get(member_index)
                definition = definition or definitions_array.get(member_index)
                definition = definition or definitions_struct.get(member_index)
                definitions.append(definition.copy())

            struct_name, member_names = self.byte_code.find_names(index)
            definitions_struct[index] = Struct(definitions, default_layout, member_names=member_names,
                                               type_name=struct_name)

        else:
            raise ByteCodeError.unexpected()

    def build_matrix_definition(self, matrix_index, layout, last_struct_index):
        # we assume that matrices always have the order decoration in bytecode (if they are part of a interface block)
        if last_struct_index is None:
            raise ByteCodeError.unexpected()

        member_indices = self.byte_code.find_member_ids(last_struct_index)
        member_orders = self.byte_code.find_orders(last_struct_index)

        # matrix is direct member of the last struct
        if matrix_index in member_indices:
            member = member_indices.index(matrix_index)

        # if the matrix is a member of an array which is a struct member, the array is also decorated with the
        # matrix order
        else:
            member = None

            for index in member_indices:
                if index in self.byte_code.types_array:
                    type_index, _ = self.byte_code.types_array[index]
                    if type_index == matrix_index:
                        member = member_indices.index(index)
                        break

            if member is None:
                raise ByteCodeError.unexpected()

        order_decoration = member_orders[member]
        order = {Decoration.ROW_MAJOR: Order.ROW_MAJOR, Decoration.COL_MAJOR: Order.COLUMN_MAJOR}[order_decoration]

        dtype, rows, cols = self.byte_code.types_matrix[matrix_index]
        return Matrix(cols, rows, dtype, layout, order)

    def deduce_layout(self, binding):
        index, usage = self.get_block_index(binding)
        definition = self.get_block_definition(binding)

        definition.layout = Layout.STD140
        match_std140 = self.check_layout(index)

        definition.layout = Layout.STD430
        match_std430 = self.check_layout(index)

        # deduce layout
        if match_std140 and not match_std430:
            definition.layout = Layout.STD140

        elif not match_std140 and match_std430:
            definition.layout = Layout.STD430

        elif match_std140 and match_std430:
            # std430 is not allowed for uniform buffer objects
            if usage == BufferUsage.UNIFORM_BUFFER:
                definition.layout = Layout.STD140
            else:
                definition.layout = Layout.STDXXX

        else:
            possible_reasons = [
                "a memory layout other than std140 or std430 was used",
                "an offset was defined manually, e.g. for a struct member: layout(offset=128) int member;",
                "a matrix memory order was defined manually, e.g. for a struct member: layout(row_major) "
                "StructXYZ structWithMatrices;"
            ]
            raise RuntimeError("Found unexpected memory offsets, this might occur because of\n" +
                               "".join(["* {}\n".format(reason) for reason in possible_reasons]))

    def check_layout(self, index):
        definition = self.definitions_block[index]

        member_indices = self.byte_code.find_member_ids(index)
        offsets_bytecode = self.byte_code.find_offsets(index)
        offsets_bytecode = [offsets_bytecode.get(i) for i in range(len(definition.definitions))]

        if None in offsets_bytecode:
            raise ByteCodeError.unexpected()

        if offsets_bytecode != definition.offsets():
            return False

        for i, (member_index, d) in enumerate(zip(member_indices, definition.definitions)):
            if isinstance(d, Array):
                if self.byte_code.find_strides(member_index) != d.strides():
                    return False

            if isinstance(d, Matrix):
                if self.byte_code.find_matrix_stride(index, i) != d.stride():
                    return False

        return True

    def get_bindings(self):
        bindings = []

        for index in self.block_data:
            bindings.append(self.block_data[index][2])

        for binding in bindings:
            count = bindings.count(binding)
            if count > 1:
                raise LavaError("Binding {} is used {} times".format(binding, count))

        return list(sorted(bindings))

    def get_block_index(self, binding):
        for index in self.block_data:
            block_type, storage_class, binding_id = self.block_data[index]

            if binding_id == binding:
                usage = None

                if block_type == Decoration.BLOCK and storage_class == StorageClass.UNIFORM:
                    usage = BufferUsage.UNIFORM_BUFFER
                elif block_type == Decoration.BUFFER_BLOCK and storage_class == StorageClass.UNIFORM:
                    usage = BufferUsage.STORAGE_BUFFER

                return index, usage

        raise ValueError("Binding {} not found".format(binding))

    def get_block(self, binding):
        index, usage = self.get_block_index(binding)
        return self.definitions_block[index], usage

    def get_block_definition(self, binding):
        return self.get_block(binding)[0]

    def get_block_usage(self, binding):
        return self.get_block(binding)[1]

    def get_block_access(self, binding):
        index, usage = self.get_block_index(binding)
        block_definition, _ = self.get_block(binding)

        if usage == BufferUsage.UNIFORM_BUFFER:
            return Access.READ_ONLY

        decorations = self.byte_code.find_accesses(index)

        if len(decorations) not in (0, len(block_definition.definitions)):  # 0 for no access decorations at all
            raise ByteCodeError.unexpected()

        accesses = set()

        for member, d in decorations.items():
            if Decoration.NON_WRITABLE in d and Decoration.NON_READABLE not in d:
                accesses.add(Access.READ_ONLY)
            elif Decoration.NON_WRITABLE not in d and Decoration.NON_READABLE in d:
                accesses.add(Access.WRITE_ONLY)
            elif Decoration.NON_WRITABLE in d and Decoration.NON_READABLE in d:
                accesses.add(Access.NEITHER)
            else:
                accesses.add(Access.READ_WRITE)

        if len(accesses) == 0:
            accesses.add(Access.READ_WRITE)
        elif len(accesses) > 1:
            raise ByteCodeError.unexpected()

        return accesses.pop()



