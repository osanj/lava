# -*- coding: UTF-8 -*-

import logging

import vulkan as vk

from lava.api.bytecode import ByteCode
from lava.api.bytes import Array, ByteRepresentation, Scalar, Struct, Vector
from lava.api.constants.spirv import Decoration, StorageClass, ExecutionModel
from lava.api.constants.vk import BufferUsage

logger = logging.getLogger(__name__)


class Shader(object):

    def __init__(self, device, path, entry_point=None):
        self.device = device
        with open(path, "rb") as f:
            self.bytez = f.read()
            self.handle = vk.vkCreateShaderModule(
                self.device.handle, vk.VkShaderModuleCreateInfo(codeSize=len(self.bytez), pCode=self.bytez), None)

        self.byte_code = ByteCode(self.bytez)

        # placeholder for inspection variables
        self.definitions_scalar = None
        self.definitions_vector = None
        self.definitions_matrix = None
        self.definitions_array = None
        self.definitions_struct = None

        # check / set entry point
        self.entry_point = self.check_entry_point(entry_point)

    def __del__(self):
        vk.vkDestroyShaderModule(self.device.handle, self.handle, None)

    def check_entry_point(self, entry_point):
        entry_points_detected = self.byte_code.find_entry_points(execution_model=ExecutionModel.GL_COMPUTE)

        if len(entry_points_detected) == 0:
            raise RuntimeError("Could not find entry points for execution model {}".format(ExecutionModel.GL_COMPUTE))

        if entry_point is not None and entry_point not in entry_points_detected:
            raise RuntimeError("Could find entry point {} in detected entry points {}".format(
                entry_point, ", ".join(entry_points_detected)))

        if entry_point is None:
            if len(entry_points_detected) > 1:
                raise RuntimeError("Multiple entry points found {}".format(", ".join(entry_points_detected)))
            entry_point = entry_points_detected[0]

        return entry_point

    def get_entry_point(self):
        return self.entry_point

    def inspect(self):
        self.inspect_definitons()
        self.inspect_layouts()

    def inspect_definitons(self):
        default_layout = ByteRepresentation.LAYOUT_STD140
        default_order = ByteRepresentation.ORDER_COLUMN_MAJOR

        self.definitions_scalar = {index: Scalar.of(dtype) for index, dtype in self.byte_code.types_scalar.items()}
        self.definitions_vector = {index: Vector(n, dtype) for index, (dtype, n) in self.byte_code.types_vector.items()}
        self.definitions_matrix = {}
        self.definitions_array = {}
        self.definitions_struct = {}

        candidates_array = list(self.byte_code.types_array.keys())
        candidates_struct = list(self.byte_code.types_struct.keys())

        while len(candidates_array) > 0 or len(candidates_struct) > 0:
            for index in candidates_array:
                type_index, dims = self.byte_code.types_array[index]

                # skip array of undefined struct
                if type_index in self.byte_code.types_struct and type_index not in self.definitions_struct:
                    break

                definition = self.definitions_scalar.get(type_index, None)
                definition = definition or self.definitions_vector.get(type_index, None)
                definition = definition or self.definitions_matrix.get(type_index, None)
                definition = definition or self.definitions_struct.get(type_index, None)

                self.definitions_array[index] = Array(definition, dims, default_layout)
                candidates_array.remove(index)

            for index in candidates_struct:
                member_indices = self.byte_code.types_struct[index]

                skip = False
                for member_index in member_indices:
                    is_struct = member_index in self.byte_code.types_struct
                    is_array = member_index in self.byte_code.types_array

                    # skip undefined struct
                    if is_struct and member_index not in self.definitions_struct:
                        skip = True
                        break

                    # skip array of undefined struct
                    if is_array and member_index not in self.definitions_array:
                        skip = True
                        break

                if skip:
                    continue

                definitions = []
                for member_index in member_indices:
                    definition = self.definitions_scalar.get(member_index, None)
                    definition = definition or self.definitions_vector.get(member_index, None)
                    definition = definition or self.definitions_matrix.get(member_index, None)
                    definition = definition or self.definitions_array.get(member_index, None)
                    definition = definition or self.definitions_struct.get(member_index, None)
                    definitions.append(definition)

                struct_name, member_names = self.byte_code.find_names(index)
                self.definitions_struct[index] = Struct(definitions, default_layout, member_names=member_names,
                                                        type_name=struct_name)

                candidates_struct.remove(index)

    def inspect_layouts(self):
        bindings = self.get_bindings()

        for binding in bindings:
            index, usage = self.get_block_index(binding)
            definition = self.definitions_struct[index]

            print ""
            print ""
            print "binding", binding
            print "index", index
            print definition
            print ""

            offsets_bytecode = self.byte_code.find_offsets(index)

            self.set_layout(index, ByteRepresentation.LAYOUT_STD140)
            offsets_std140 = definition.offsets()
            match_std140 = offsets_bytecode == definition.offsets()

            self.set_layout(index, ByteRepresentation.LAYOUT_STD430)
            offsets_std430 = definition.offsets()
            match_std430 = offsets_bytecode == definition.offsets()

            if match_std140 and not match_std430:
                self.set_layout(index, ByteRepresentation.LAYOUT_STD140)
            elif not match_std140 and match_std430:
                self.set_layout(index, ByteRepresentation.LAYOUT_STD430)
            elif match_std140 and match_std430:
                self.set_layout(index, ByteRepresentation.LAYOUT_STDXXX)
            else:
                print "bytecode", offsets_bytecode
                print "std140  ", offsets_std140
                print "std430  ", offsets_std430
                raise RuntimeError("Found unexpected memory offsets")

            print index, definition.layout

    def set_layout(self, index, layout):
        if index in self.definitions_struct:
            member_indices = self.byte_code.types_struct[index]

            for member_index in member_indices:
                if member_index in self.byte_code.types_struct or member_index in self.byte_code.types_array:
                    self.set_layout(member_index, layout)

            self.definitions_struct[index].layout = layout  # set after setting children!

        elif index in self.definitions_array:
            type_index, _ = self.byte_code.types_array[index]

            if type_index in self.byte_code.types_struct:
                self.set_layout(type_index, layout)

            self.definitions_array[index].layout = layout  # set after setting children!s

        else:
            raise RuntimeError()

    def get_bindings(self):
        block_data = self.byte_code.find_blocks()
        bindings = []

        for index in block_data:
            bindings.append(block_data[index][2])

        return list(sorted(bindings))

    def get_block_index(self, binding):
        block_data = self.byte_code.find_blocks()

        for index in block_data:
            block_type, storage_class, binding_id = block_data[index]

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
        return self.definitions_struct[index], usage
