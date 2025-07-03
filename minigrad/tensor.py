from typing import Union, Any
import numbers
import numpy as np


class NameManager:
    _counts = {}

    @staticmethod
    def reset():
        NameManager._counts = {}

    @staticmethod
    def _count(name):
        if name not in NameManager._counts:
            NameManager._counts[name] = 0
        count = NameManager._counts[name]
        return count

    @staticmethod
    def _inc_count(name):
        assert name in NameManager._counts, f'Name {name} is not registered.'
        NameManager._counts[name] += 1

    @staticmethod
    def new(name: str):
        count = NameManager._count(name)
        tensor_name = f"{name}:{count}"
        NameManager._inc_count(name)
        return tensor_name
    

class Tensor:
    """Classe representando um array multidimensional.

    Atributos:

    - _arr  (privado): dados internos do tensor como
        um array do numpy com 2 dimensões (ver Regras)

    - _parents (privado): lista de tensores que foram
        usados como argumento para a operação que gerou o
        tensor. Será vazia se o tensor foi inicializado com
        valores diretamente. Por exemplo, se o tensor foi
        resultado da operação a + b entre os tensores a e b,
        _parents = [a, b].

    - requires_grad (público): indica se devem ser
        calculados gradientes para o tensor ou não.

    - grad (público): Tensor representando o gradiente.

    """

    def __init__(self,
                 # Dados do tensor. Além dos tipos listados,
                 # arr também pode ser do tipo Tensor.
                 arr: Union[np.ndarray, list, numbers.Number, Any],
                 # Entradas da operacao que gerou o tensor.
                 # Deve ser uma lista de itens do tipo Tensor.
                 parents: list[Any] = [],
                 # se o tensor requer o calculo de gradientes ou nao
                 requires_grad: bool = True,
                 # nome do tensor
                 name: str = '',
                 # referência para um objeto do tipo Operation (ou
                 # subclasse) indicando qual operação gerou este
                 # tensor. Este objeto também possui um método
                 # para calcular a derivada da operação.
                 operation=None):
        
        if isinstance(arr, Tensor):
            self._arr = arr.numpy().copy()
        else:
            self._arr = np.array(arr, dtype=float)

        if self._arr.ndim == 0:
            self._arr = self._arr.reshape(1, 1)
        elif self._arr.ndim == 1:
            self._arr = self._arr.reshape(-1, 1)
        elif self._arr.ndim > 2:
            raise ValueError('A dimensão do array não pode ser maior que 2.')

        self.requires_grad = requires_grad
        self._parents = parents
        self._operation = operation
        self._name = name if name else NameManager.new('in')
        self.grad = None

    @property
    def shape(self):
        return self._arr.shape # dessa forma não precisamos ficar usando .numpy().shape

    def zero_grad(self):
        """Reinicia o gradiente com zero"""
        self.grad = Tensor(np.zeros_like(self._arr), requires_grad=False)

    def numpy(self):
        """Retorna o array interno"""
        return self._arr

    def __repr__(self):
        """Permite visualizar os dados do tensor como string"""
        return f"Tensor({self.numpy()}, name={self._name}, shape={self.shape})"

    def backward(self, my_grad=None):
        """Método usado tanto iniciar o processo de
        diferenciação automática, quanto por um filho
        para enviar o gradiente do pai. No primeiro
        caso, o argumento my_grad não será passado.
        """
        # Verificar se o tensor requer gradiente
        if not self.requires_grad:
            return

        if my_grad is None:
            # Início do backprop, gradiente da saída em relação a si mesma é 1.
            my_grad = Tensor(np.ones_like(self.numpy()), requires_grad=False)

        if self.grad is None:
            self.grad = Tensor(my_grad.numpy(), requires_grad=False)
        else:
            # somar gradientes no nível do NumPy
            current_grad_arr = self.grad.numpy()
            incoming_grad_arr = my_grad.numpy()
            self.grad = Tensor(current_grad_arr + incoming_grad_arr, requires_grad=False)

        if self._operation:
            parent_grads = self._operation.grad(my_grad, *self._parents)
            for parent, grad in zip(self._parents, parent_grads):
                parent.backward(grad)

    # sobrecarga de operadores
    def __add__(self, other):
        """Sobrecarga do operador de adição (+)"""
        from .ops import add
        return add(self, other)

    def __radd__(self, other):
        """Sobrecarga do operador de adição para o caso other + self"""
        from .ops import add
        return add(other, self)

    def __sub__(self, other):
        """Sobrecarga do operador de subtração (-)"""
        from .ops import sub
        return sub(self, other)

    def __rsub__(self, other):
        """Sobrecarga do operador de subtração para o caso other - self"""
        from .ops import sub
        return sub(other, self)

    def __mul__(self, other):
        """Sobrecarga do operador de multiplicação (*)"""
        from .ops import prod
        return prod(self, other)

    def __rmul__(self, other):
        """Sobrecarga do operador de multiplicação para o caso other * self"""
        from .ops import prod
        return prod(other, self)

    def __matmul__(self, other):
        """Sobrecarga do operador de multiplicação de matriz (@)"""
        from .ops import matmul
        return matmul(self, other)

    def __rmatmul__(self, other):
        """Sobrecarga do operador de multiplicação de matriz para o caso other @ self"""
        from .ops import matmul
        return matmul(other, self)

    def __neg__(self):
        """Sobrecarga do operador de negação (-)"""
        from .ops import prod
        return prod(self, -1)

    def __pow__(self, exponent):
        """Sobrecarga do operador de potência (**)"""
        from .ops import power
        return power(self, exponent)