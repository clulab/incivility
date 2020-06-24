# taken from:
# https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L647

import tensorflow as tf
import tensorflow.keras.backend as K


def insert_arguments(**arguments):
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


def export_to_custom_objects(base_extend_with):
    def new_extend_with(BaseOptimizer, name=None):
        NewOptimizer = base_extend_with(BaseOptimizer)

        if isinstance(name, str):
            NewOptimizer.__name__ = name

        name = NewOptimizer.__name__
        tf.keras.utils.get_custom_objects()[name] = NewOptimizer

        return NewOptimizer

    return new_extend_with


@export_to_custom_objects
def extend_with_gradient_accumulation_v2(BaseOptimizer):
    class NewOptimizer(BaseOptimizer):
        @insert_arguments(grad_accum_steps=2)
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def _create_slots(self, var_list):
            super(NewOptimizer, self)._create_slots(var_list)
            for var in var_list:
                self.add_slot(var, 'ag')

        def _resource_apply(self, grad, var, indices=None):
            cond = K.equal(self.iterations % self.grad_accum_steps, 0)
            ag = self.get_slot(var, 'ag')

            old_update = K.update

            def new_update(x, new_x):
                new_x = K.switch(cond, new_x, x)
                return old_update(x, new_x)

            K.update = new_update
            ag_t = ag / self.grad_accum_steps
            op = super(NewOptimizer, self)._resource_apply(ag_t, var)
            K.update = old_update

            with tf.control_dependencies([op]):
                ag_t = K.switch(cond, K.zeros_like(ag), ag)
                with tf.control_dependencies([K.update(ag, ag_t)]):
                    if indices is None:
                        ag_t = K.update(ag, ag + grad)
                    else:
                        ag_t = self._resource_scatter_add(ag, indices, grad)

            return ag_t

        def get_config(self):
            config = {
                'grad_accum_steps': self.grad_accum_steps,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


AdamGA = extend_with_gradient_accumulation_v2(tf.optimizers.Adam)
