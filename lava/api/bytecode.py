# -*- coding: UTF-8 -*-

import itertools
import struct

import lava.api.constants.spirv as spirv
from lava.api.util import LavaError


class ByteCodeError(LavaError):

    UNEXPECTED = "Something unexpected happened"

    def __init__(self, message):
        super(ByteCodeError, self).__init__(message)

    @classmethod
    def unexpected(cls):
        return cls(cls.UNEXPECTED)


class ByteCode(object):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf

    def __init__(self, bytez):
        self.bytez = bytearray(bytez)

        self.header = ByteCodeHeader(self.bytez)
        self.instructions = []

        # parse instruction by instruction
        step = self.header.words() * spirv.WORD_BYTE_SIZE
        while step < len(bytez):
            instruction = ByteCodeInstruction(self.bytez[step:])
            self.instructions.append(instruction)
            step += instruction.words() * spirv.WORD_BYTE_SIZE

        # recover types
        self.types_scalar, self.types_vector, self.types_matrix = self.find_basic_types()
        self.types_array = self.find_array_types()
        self.types_struct = self.find_struct_types()

    @classmethod
    def read_word(cls, bytez, offset=0):
        # word to unsigned integer
        return cls.read_words(bytez, n=1, offset=offset)[0]

    @classmethod
    def read_words(cls, bytez, n=-1, offset=0):
        # words to unsigned integers
        if n == -1:
            n = len(bytez) // spirv.WORD_BYTE_SIZE - offset
        a = offset * spirv.WORD_BYTE_SIZE
        b = a + n * spirv.WORD_BYTE_SIZE
        return struct.unpack("I" * n, bytez[a:b])

    @classmethod
    def read_words_as_string(cls, bytez, n=-1, offset=0):
        a = offset * spirv.WORD_BYTE_SIZE
        b = len(bytez) if n == -1 else a + n * spirv.WORD_BYTE_SIZE
        return bytez[a:b].rstrip(b"\0").decode("utf-8")

    def __str__(self):
        strings = []
        for instruction in self.instructions:
            if instruction.op is None:
                strings.append("Op id={} words={}".format(instruction.op_id, instruction.words()))
            else:
                strings.append(str(instruction.op))
        return "\n".join(strings)

    def abort(self):
        raise ByteCodeError.unexpected()

    def find_instructions(self, operation):
        results = []
        for instruction in self.instructions:
            if instruction.op_id == operation.ID:
                results.append(instruction)
        return results

    def find_instructions_with_attributes(self, operation, **attributes):
        results = self.find_instructions(operation)
        results_filtered = []

        for instruction in results:
            matches = 0

            for attr_key, attr_value in attributes.items():
                if attr_key not in instruction.op.__dict__:
                    break
                if instruction.op.__dict__[attr_key] != attr_value:
                    break
                matches += 1

            if matches == len(attributes):
                results_filtered.append(instruction)

        return results_filtered

    def find_basic_types(self):
        types_scalar = {}
        types_vector = {}
        types_matrix = {}

        search_scalar = [
            (spirv.DataType.FLOAT, {"operation": OpTypeFloat, "width": 32}),
            (spirv.DataType.DOUBLE, {"operation": OpTypeFloat, "width": 64}),
            (spirv.DataType.INT, {"operation": OpTypeInt, "width": 32, "signedness": 1}),
            (spirv.DataType.UINT, {"operation": OpTypeInt, "width": 32, "signedness": 0}),
        ]

        # scalar types are "standalone"
        for type_scalar, search_data in search_scalar:
            instructions = self.find_instructions_with_attributes(**search_data)
            if len(instructions) == 1:
                types_scalar[instructions[0].op.result_id] = type_scalar
            elif len(instructions) > 1:
                self.abort()

        # vector types reference the scalar types
        for result_id, n in itertools.product(types_scalar.keys(), range(2, 5)):
            instructions = self.find_instructions_with_attributes(operation=OpTypeVector, component_type=result_id,
                                                                  component_count=n)
            if len(instructions) == 1:
                types_vector[instructions[0].op.result_id] = (types_scalar[result_id], n)
            elif len(instructions) > 1:
                self.abort()

        # matrix types reference the vector types
        for result_id, cols in itertools.product(types_vector.keys(), range(2, 5)):
            instructions = self.find_instructions_with_attributes(operation=OpTypeMatrix, column_type=result_id,
                                                                  column_count=cols)
            if len(instructions) == 1:
                scalar_type, rows = types_vector[result_id]
                types_matrix[instructions[0].op.result_id] = (scalar_type, rows, cols)
            elif len(instructions) > 1:
                self.abort()

        return types_scalar, types_vector, types_matrix

    def find_array_types(self):
        types_array = {}
        tmp = {}

        # find all array types first
        for instruction in self.find_instructions(OpTypeArray):
            constants = self.find_instructions_with_attributes(OpConstant, result_id=instruction.op.length)
            if len(constants) == 1:
                n = constants[0].op.literals[0]
                tmp[instruction.op.result_id] = (instruction.op.element_type, [n])

        # collapse them into nd-arrays
        for idx in sorted(tmp.keys(), reverse=True):
            if idx in tmp:
                ref, dims = tmp[idx]

                while ref in tmp:
                    other_ref, other_dims = tmp[ref]
                    del tmp[ref]
                    ref = other_ref
                    dims += other_dims

                types_array[idx] = (ref, tuple(dims))
                del tmp[idx]

        return types_array

    def find_struct_types(self):
        types_struct = {}

        for instruction in self.find_instructions(OpTypeStruct):
            types_struct[instruction.op.result_id] = instruction.op.member_types

        return types_struct

    def find_names(self, struct_id):
        struct_name = None

        instructions = self.find_instructions_with_attributes(OpName, target_id=struct_id)
        if len(instructions) == 1:
            struct_name = instructions[0].op.name

        instructions = self.find_instructions_with_attributes(OpMemberName, type_id=struct_id)
        member_names = {instruction.op.member: instruction.op.name for instruction in instructions}
        return struct_name, [member_names.get(i) for i in range(max(member_names.keys()) + 1)]

    def find_member_ids(self, struct_id):
        instructions = self.find_instructions_with_attributes(OpTypeStruct, result_id=struct_id)
        if len(instructions) == 1:
            return instructions[0].op.member_types
        else:
            return None

    def find_offsets(self, struct_id):
        offsets = {}

        for instruction in self.find_instructions_with_attributes(OpMemberDecorate, decoration=spirv.Decoration.OFFSET,
                                                                  type_id=struct_id):
            member = instruction.op.member
            offset = instruction.op.literals[0]
            offsets[member] = offset

        return offsets

    def find_accesses(self, struct_id):
        accesses = {}

        instructions1 = self.find_instructions_with_attributes(OpMemberDecorate, type_id=struct_id,
                                                               decoration=spirv.Decoration.NON_READABLE)
        instructions2 = self.find_instructions_with_attributes(OpMemberDecorate, type_id=struct_id,
                                                               decoration=spirv.Decoration.NON_WRITABLE)

        for instruction in instructions1 + instructions2:
            member = instruction.op.member

            tmp = accesses.get(member, set())
            tmp.add(instruction.op.decoration)
            accesses[member] = tmp

        return accesses

    def find_strides(self, array_id):
        array_ids = [array_id]

        instructions = self.find_instructions_with_attributes(OpTypeArray, result_id=array_ids[-1])
        while len(instructions) == 1:
            array_ids.append(instructions[0].op.element_type)
            instructions = self.find_instructions_with_attributes(OpTypeArray, result_id=array_ids[-1])
        array_ids = array_ids[:-1]  # last type is no array

        strides = []

        for array_id in array_ids:
            instructions = self.find_instructions_with_attributes(OpDecorate, decoration=spirv.Decoration.ARRAY_STRIDE,
                                                                  target_id=array_id)
            if len(instructions) == 1:
                strides.append(instructions[0].op.literals[0])
            else:
                self.abort()

        return strides

    def find_matrix_stride(self, struct_id, member):
        instructions = self.find_instructions_with_attributes(OpMemberDecorate, type_id=struct_id, member=member,
                                                              decoration=spirv.Decoration.MATRIX_STRIDE)
        if len(instructions) == 1:
            return instructions[0].op.literals[0]
        else:
            return None

    def find_orders(self, struct_id):
        orders = {}

        instructions_row_major = self.find_instructions_with_attributes(OpMemberDecorate, type_id=struct_id,
                                                                        decoration=spirv.Decoration.ROW_MAJOR)

        instructions_col_major = self.find_instructions_with_attributes(OpMemberDecorate, type_id=struct_id,
                                                                        decoration=spirv.Decoration.COL_MAJOR)

        for instruction in instructions_row_major + instructions_col_major:
            member = instruction.op.member
            orders[member] = instruction.op.decoration

        return orders

    def find_blocks(self):
        blocks = {}
        blocks1 = self.find_instructions_with_attributes(OpDecorate, decoration=spirv.Decoration.BLOCK)  # ubo
        blocks2 = self.find_instructions_with_attributes(OpDecorate, decoration=spirv.Decoration.BUFFER_BLOCK)  # ssbo

        for candidate in blocks1 + blocks2:
            # find associated pointer
            instructions = self.find_instructions_with_attributes(OpTypePointer, type_id=candidate.op.target_id)
            if len(instructions) != 1:
                self.abort()

            # find associated variable
            instructions = self.find_instructions_with_attributes(OpVariable, result_type=instructions[0].op.result_id)
            if len(instructions) != 1:
                self.abort()
            storage_class = instructions[0].op.storage_class

            # find associated binding
            instructions = self.find_instructions_with_attributes(OpDecorate, target_id=instructions[0].op.result_id,
                                                                  decoration=spirv.Decoration.BINDING)

            block_type = candidate.op.decoration

            if len(instructions) == 1:
                binding_id = instructions[0].op.literals[0]
                blocks[candidate.op.target_id] = (block_type, storage_class, binding_id)

            else:
                self.abort()

        return blocks

    def find_entry_points(self, execution_model):
        entry_points = {}
        instructions = self.find_instructions_with_attributes(OpEntryPoint, execution_model=execution_model)

        for instruction in instructions:
            entry_points[instruction.op.entry_point] = instruction.op.name

        return entry_points

    def find_entry_point_details(self, entry_point_index):
        execution_mode = None
        literals = None
        instructions = self.find_instructions_with_attributes(OpExecutionMode, entry_point=entry_point_index)

        if len(instructions) == 1:
            execution_mode = instructions[0].op.execution_mode
            literals = instructions[0].op.literals
        else:
            self.abort()

        return execution_mode, literals


class ByteCodeHeader(object):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#subsection.2.3

    def __init__(self, bytez):
        self.bytez = bytez
        magic_number, version, generator_magic_number, bound, _ = ByteCode.read_words(bytez, spirv.WORD_COUNT_HEADER)

        if magic_number != spirv.MAGIC_NUMBER:
            raise ByteCodeError("MagicNumber does not match SPIR-V specs")

        self.version_major = (version & 0x00FF0000) >> 16
        self.version_minor = (version & 0x0000FF00) >> 8
        self.generator_magic_number = generator_magic_number
        self.bound = bound

    def words(self):
        return spirv.WORD_COUNT_HEADER


class ByteCodeInstruction(object):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#subsection.2.3

    def __init__(self, bytez):
        first_word = ByteCode.read_word(bytez)
        self.word_count = first_word >> 16
        self.op_id = first_word & 0x0000FFFF
        self.bytez = bytez[:self.word_count * spirv.WORD_BYTE_SIZE]
        self.op = None

        if self.op_id in OPS_REGISTER:
            self.op = OPS_REGISTER[self.op_id](self.bytez[spirv.WORD_BYTE_SIZE:])

    def words(self):
        return self.word_count


class Op(object):
    ID = -1

    def __init__(self, bytez):
        self.bytez = bytez

    def describe(self):
        raise NotImplementedError()

    def __str__(self):
        return "{:<20}{}".format(self.__class__.__name__, self.describe() or "")


class OpName(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpName
    ID = 5

    def __init__(self, bytez):
        super(OpName, self).__init__(bytez)
        self.target_id = ByteCode.read_word(bytez)
        self.name = ByteCode.read_words_as_string(bytez, offset=1)

    def describe(self):
        return "target_id={} name={}".format(self.target_id, self.name)


class OpMemberName(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpMemberName
    ID = 6

    def __init__(self, bytez):
        super(OpMemberName, self).__init__(bytez)
        self.type_id = ByteCode.read_word(bytez)
        self.member = ByteCode.read_word(bytez, offset=1)
        self.name = ByteCode.read_words_as_string(bytez, offset=2)

    def describe(self):
        return "type_id={} member_id={} name={}".format(self.type_id, self.member, self.name)


class OpEntryPoint(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpEntryPoint
    ID = 15

    def __init__(self, bytez):
        super(OpEntryPoint, self).__init__(bytez)
        self.execution_model = spirv.ExecutionModel.from_spirv(ByteCode.read_word(bytez))
        self.entry_point = ByteCode.read_word(bytez, offset=1)
        self.name = ByteCode.read_words_as_string(bytez, n=1, offset=2)
        self.ids = ByteCode.read_words(bytez, offset=3)

    def describe(self):
        return "execution_model={} entry_point={} name={} ids={}".format(self.execution_model, self.entry_point,
                                                                         self.name, self.ids)


class OpExecutionMode(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpExecutionMode
    ID = 16

    def __init__(self, bytez):
        super(OpExecutionMode, self).__init__(bytez)
        self.entry_point = ByteCode.read_word(bytez)
        self.execution_mode = spirv.ExecutionMode.from_spirv(ByteCode.read_word(bytez, offset=1))
        self.literals = ByteCode.read_words(bytez, offset=2)

    def describe(self):
        return "entry_point={} execution_mode={} literals={}".format(self.entry_point, self.execution_mode,
                                                                     self.literals)


class OpDecorate(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpDecorate
    ID = 71

    def __init__(self, bytez):
        super(OpDecorate, self).__init__(bytez)
        self.target_id = ByteCode.read_word(bytez)
        self.decoration = spirv.Decoration.from_spirv(ByteCode.read_word(bytez, offset=1))
        self.literals = ByteCode.read_words(bytez, offset=2)

    def describe(self):
        return "target_id={} decoration={} literals={}".format(self.target_id, self.decoration, self.literals)


class OpMemberDecorate(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpMemberDecorate
    ID = 72

    def __init__(self, bytez):
        super(OpMemberDecorate, self).__init__(bytez)
        self.type_id = ByteCode.read_word(bytez)
        self.member = ByteCode.read_word(bytez, offset=1)
        self.decoration = spirv.Decoration.from_spirv(ByteCode.read_word(bytez, offset=2))
        self.literals = ByteCode.read_words(bytez, n=(len(bytez) // spirv.WORD_BYTE_SIZE - 3), offset=3)

    def describe(self):
        return "type_id={} member={} decoration={} literals={}".format(self.type_id, self.member, self.decoration,
                                                                       self.literals)


class OpTypeVoid(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpTypeVoid
    ID = 19

    def __init__(self, bytez):
        super(OpTypeVoid, self).__init__(bytez)
        self.result_id = ByteCode.read_word(bytez)

    def describe(self):
        return "result_id={}".format(self.result_id)


class OpTypeBool(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpTypeBool
    ID = 20

    def __init__(self, bytez):
        super(OpTypeBool, self).__init__(bytez)
        self.result_id = ByteCode.read_word(bytez)

    def describe(self):
        return "result_id={}".format(self.result_id)


class OpTypeInt(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpTypeInt
    ID = 21

    def __init__(self, bytez):
        super(OpTypeInt, self).__init__(bytez)
        self.result_id = ByteCode.read_word(bytez)
        self.width = ByteCode.read_word(bytez, offset=1)
        self.signedness = ByteCode.read_word(bytez, offset=2)

    def describe(self):
        return "result_id={} width={} signedness={}".format(self.result_id, self.width, self.signedness)


class OpTypeFloat(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpTypeFloat
    ID = 22

    def __init__(self, bytez):
        super(OpTypeFloat, self).__init__(bytez)
        self.result_id = ByteCode.read_word(bytez)
        self.width = ByteCode.read_word(bytez, offset=1)

    def describe(self):
        return "result_id={} width={}".format(self.result_id, self.width)


class OpTypeVector(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpTypeVector
    ID = 23

    def __init__(self, bytez):
        super(OpTypeVector, self).__init__(bytez)
        self.result_id = ByteCode.read_word(bytez)
        self.component_type = ByteCode.read_word(bytez, offset=1)
        self.component_count = ByteCode.read_word(bytez, offset=2)

    def describe(self):
        return "result_id={} component_type={} component_count={}".format(self.result_id, self.component_type,
                                                                          self.component_count)


class OpTypeMatrix(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpTypeMatrix
    ID = 24

    def __init__(self, bytez):
        super(OpTypeMatrix, self).__init__(bytez)
        self.result_id = ByteCode.read_word(bytez)
        self.column_type = ByteCode.read_word(bytez, offset=1)
        self.column_count = ByteCode.read_word(bytez, offset=2)

    def describe(self):
        return "result_id={} column_type={} column_count={}".format(self.result_id, self.column_type, self.column_count)


class OpTypeImage(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpTypeImage
    ID = 25

    def __init__(self, bytez):
        super(OpTypeImage, self).__init__(bytez)
        self.result_id = ByteCode.read_word(bytez)
        self.sampled_type = ByteCode.read_word(bytez, offset=1)
        # TODO: read out remaining 5 tons of information

    def describe(self):
        return "result_id={} sampled_type={}".format(self.result_id, self.sampled_type)


class OpTypeSampler(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpTypeSampler
    ID = 26

    def __init__(self, bytez):
        super(OpTypeSampler, self).__init__(bytez)
        self.result_id = ByteCode.read_word(bytez)

    def describe(self):
        return "result_id={}".format(self.result_id)


class OpTypeArray(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpTypeArray
    ID = 28

    def __init__(self, bytez):
        super(OpTypeArray, self).__init__(bytez)
        self.result_id = ByteCode.read_word(bytez)
        self.element_type = ByteCode.read_word(bytez, offset=1)
        self.length = ByteCode.read_word(bytez, offset=2)

    def describe(self):
        return "result_id={} element_type={} length={}".format(self.result_id, self.element_type, self.length)


class OpTypeRuntimeArray(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpTypeRuntimeArray
    ID = 29

    def __init__(self, bytez):
        super(OpTypeRuntimeArray, self).__init__(bytez)
        self.result_id = ByteCode.read_word(bytez)
        self.element_type = ByteCode.read_word(bytez, offset=1)

    def describe(self):
        return "result_id={} element_type={}".format(self.result_id, self.element_type)


class OpTypeStruct(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpTypeStruct
    ID = 30

    def __init__(self, bytez):
        super(OpTypeStruct, self).__init__(bytez)
        self.result_id = ByteCode.read_word(bytez)
        self.member_types = ByteCode.read_words(bytez, offset=1)

    def describe(self):
        return "result_id={} member_types={}".format(self.result_id, self.member_types)


class OpSource(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpSource
    ID = 3

    def __init__(self, bytez):
        super(OpSource, self).__init__(bytez)
        self.source_language = spirv.SourceLanguage.from_spirv(ByteCode.read_word(bytez))
        self.version = ByteCode.read_word(bytez, offset=1)
        # ignore other attributes

    def describe(self):
        return "source_language={} version={}".format(self.source_language, self.version)


class OpSourceExtension(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpSourceExtension
    ID = 4

    def __init__(self, bytez):
        super(OpSourceExtension, self).__init__(bytez)
        self.extension = ByteCode.read_words_as_string(bytez)

    def describe(self):
        return "extension={}".format(self.extension)


class OpTypePointer(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpTypePointer
    ID = 32

    def __init__(self, bytez):
        super(OpTypePointer, self).__init__(bytez)
        self.result_id = ByteCode.read_word(bytez)
        self.storage_class = spirv.StorageClass.from_spirv(ByteCode.read_word(bytez, offset=1))
        self.type_id = ByteCode.read_word(bytez, offset=2)

    def describe(self):
        return "result_id={} storage_class={} type_id={}".format(self.result_id, self.storage_class, self.type_id)


class OpVariable(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpVariable
    ID = 59

    def __init__(self, bytez):
        super(OpVariable, self).__init__(bytez)
        self.result_type = ByteCode.read_word(bytez)
        self.result_id = ByteCode.read_word(bytez, offset=1)
        self.storage_class = spirv.StorageClass.from_spirv(ByteCode.read_word(bytez, offset=2))
        # self.initializer = ... (optional)

    def describe(self):
        return "result_type={} result_id={} storage_class={}".format(self.result_type, self.result_id,
                                                                     self.storage_class)


class OpConstant(Op):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#OpConstant
    ID = 43

    def __init__(self, bytez):
        super(OpConstant, self).__init__(bytez)
        self.result_type = ByteCode.read_word(bytez)
        self.result_id = ByteCode.read_word(bytez, offset=1)
        self.literals = ByteCode.read_words(bytez, offset=2)
        # self.initializer = ... (optional)

    def describe(self):
        return "result_type={} result_id={} literals={}".format(self.result_type, self.result_id, self.literals)


OPS_REGISTER = {op.ID: op for op in [
    OpName, OpMemberName, OpEntryPoint, OpExecutionMode, OpDecorate, OpMemberDecorate, OpTypeBool, OpTypeInt,
    OpTypeFloat, OpTypeVector, OpTypeMatrix, OpTypeArray, OpTypeRuntimeArray, OpTypeStruct, OpTypeVoid, OpTypeImage,
    OpTypeSampler, OpSource, OpSourceExtension, OpTypePointer, OpVariable, OpConstant
]}
