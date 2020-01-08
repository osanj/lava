# -*- coding: UTF-8 -*-

from lava.api.util import LavaError


class ByteCodeError(LavaError):

    UNEXPECTED = "Something unexpected happened"

    def __init__(self, message):
        super(ByteCodeError, self).__init__(message)

    @classmethod
    def unexpected(cls):
        return cls(cls.UNEXPECTED)
