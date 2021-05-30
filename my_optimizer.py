import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer

class MyOptimizer(Optimizer):
  def _resource_apply_dense(self, grad, handle, apply_state):
    """Add ops to apply dense gradients to the variable `handle`.

    Args:
      grad: a `Tensor` representing the gradient.
      handle: a `Tensor` of dtype `resource` which points to the variable to be
        updated.
      apply_state: A dict which is used across multiple apply calls.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    raise NotImplementedError("Optimizer can not handle dense matrices.")

  def _resource_apply_sparse(self, grad, handle, indices, apply_state):
    """Add ops to apply sparse gradients to the variable `handle`.

    Similar to `_apply_sparse`, the `indices` argument to this method has been
    de-duplicated. Optimizers which deal correctly with non-unique indices may
    instead override `_resource_apply_sparse_duplicate_indices` to avoid this
    overhead.

    Args:
      grad: a `Tensor` representing the gradient for the affected indices.
      handle: a `Tensor` of dtype `resource` which points to the variable to be
        updated.
      indices: a `Tensor` of integral type representing the indices for which
        the gradient is nonzero. Indices are unique.
      apply_state: A dict which is used across multiple apply calls.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    raise NotImplementedError("Optimizer can not handle sparse matrices.")
  
  def _create_slots(self, var_list):
    """ additional variables required by this optimizer """
    pass

  def get_config(self):
    """Returns the config of the optimizer.

    An optimizer config is a Python dictionary (serializable)
    containing the configuration of an optimizer (includes all Hyperparameters).
    The same optimizer can be reinstantiated later
    (without any saved state) from this configuration.

    Returns:
        Python dictionary.
    """
    config = {"name": self._name}
    if self.clipnorm is not None:
      config["clipnorm"] = self.clipnorm
    if self.clipvalue is not None:
      config["clipvalue"] = self.clipvalue
    if self.global_clipnorm is not None:
      config["global_clipnorm"] = self.global_clipnorm
    return config