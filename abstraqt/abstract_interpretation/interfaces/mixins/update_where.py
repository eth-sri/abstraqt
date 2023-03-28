from abc import ABC

from abstraqt.abstract_interpretation.interfaces.mixins.abstract_lift_join import AbstractJoin


class UpdateWhere(AbstractJoin, ABC):

    def update_where(self, positions, new_values):
        from abstraqt.abstract_interpretation import AbstractBoolArray
        assert isinstance(positions, AbstractBoolArray)
        assert isinstance(new_values, self.__class__)

        updates = positions.equal_abstract_object_element_wise(1)
        self[updates] = new_values[updates]

        tops = positions.equal_abstract_object_element_wise(AbstractBoolArray.top(()))
        self[tops] = self[tops].join(new_values[tops])

        return self

    #####################
    # TO BE IMPLEMENTED #
    #####################

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()
