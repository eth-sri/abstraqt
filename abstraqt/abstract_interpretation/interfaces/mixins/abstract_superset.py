from abc import ABC

from abstraqt.abstract_interpretation.interfaces.mixins.abstract_lift_join import AbstractLift


class AbstractSuperSet(AbstractLift, ABC):

    def is_super_set_of(self, other):
        """
        True iff 𝛾(self) ⊇ 𝛾(other)
        """
        raise NotImplementedError()

    def contains_reference(self, reference):
        """
        True if reference (in instance of the reference implementation) lies in self
        """
        reference = self.lift(reference)
        ret = self.is_super_set_of(reference)
        return ret
