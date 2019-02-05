# -*- coding: UTF-8 -*-

import logging
import struct

import vulkan as vk

import lava.api.constants.spirv as spirv

logger = logging.getLogger(__name__)


class Shader(object):

    def __init__(self, device, path, entry_point="main"):
        self.device = device
        self.entry_point = entry_point
        with open(path, "rb") as f:
            self.byte_code = f.read()

            create_info = vk.VkShaderModuleCreateInfo(codeSize=len(self.byte_code), pCode=self.byte_code)
            self.handle = vk.vkCreateShaderModule(self.device.handle, create_info, None)

    def __del__(self):
        vk.vkDestroyShaderModule(self.device.handle, self.handle, None)

    def get_entry_point(self):
        return self.entry_point


class ByteCode(object):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf

    def __init__(self, bytez):
        self.bytez = bytearray(bytez)

        self.header = ByteCodeHeader(self.bytez)
        self.instructions = []

        step = self.header.words() * spirv.WORD_BYTE_SIZE
        while step < len(bytez):
            instruction = ByteCodeInstruction(self.bytez[step:])
            self.instructions.append(instruction)
            step += instruction.words() * spirv.WORD_BYTE_SIZE

    @classmethod
    def read_word(cls, bytez, offset=0):
        # word to unsigned integer
        return cls.read_words(bytez, n=1, offset=offset)[0]

    @classmethod
    def read_words(cls, bytez, n=-1, offset=0):
        # words to unsigned integers
        if n == -1:
            n = len(bytez) / spirv.WORD_BYTE_SIZE - offset
        a = offset * spirv.WORD_BYTE_SIZE
        b = a + n * spirv.WORD_BYTE_SIZE
        return struct.unpack("I" * n, bytez[a:b])

    @classmethod
    def read_words_as_string(cls, bytez, n=-1, offset=0):
        a = offset * spirv.WORD_BYTE_SIZE
        b = len(bytez) if n == -1 else a + n * spirv.WORD_BYTE_SIZE
        return bytez[a:b].rstrip("\0")

    def __str__(self):
        strings = []
        for instruction in self.instructions:
            if instruction.op is None:
                strings.append("Op id={} words={}".format(instruction.op_id, instruction.words()))
            else:
                strings.append(str(instruction.op))
        return "\n".join(strings)

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

            for attr_key, attr_value in attributes.iteritems():
                if not attr_key in instruction.op.__dict__:
                    break
                if instruction.op.__dict__[attr_key] != attr_value:
                    break
                matches += 1

            if matches == len(attributes):
                results_filtered.append(instruction)

        return results_filtered



    # get entry point
    # get binding
    # get std140
    # get struct layout ?


class ByteCodeHeader(object):
    # https://www.khronos.org/registry/spir-v/specs/1.2/SPIRV.pdf#subsection.2.3

    def __init__(self, bytez):
        self.bytez = bytez
        magic_number, version, generator_magic_number, bound, _ = ByteCode.read_words(bytez, spirv.WORD_COUNT_HEADER)

        if magic_number != spirv.MAGIC_NUMBER:
            raise RuntimeError("MagicNumber does not match SPIR-V specs")

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
        self.literals = ByteCode.read_words(bytez, n=(len(bytez) / spirv.WORD_BYTE_SIZE - 3), offset=3)

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


OPS_REGISTER = {op.ID: op for op in [
    OpName, OpMemberName, OpEntryPoint, OpExecutionMode, OpDecorate, OpMemberDecorate, OpTypeBool, OpTypeInt,
    OpTypeFloat, OpTypeVector, OpTypeMatrix, OpTypeArray, OpTypeRuntimeArray, OpTypeStruct, OpTypeVoid, OpTypeImage,
    OpTypeSampler, OpSource, OpSourceExtension, OpTypePointer, OpVariable
]}
