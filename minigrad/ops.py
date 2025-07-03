from .tensor import Tensor, NameManager
from abc import ABC, abstractmethod
import numpy as np

class Op(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando as entradas e
            retorna o tensor resultado. O método deve
            garantir que o atributo parents do tensor
            de saída seja uma lista de tensores."""

    @abstractmethod
    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna os gradientes dos pais em como tensores.

        Arguments:

        - back_grad: Derivada parcial em relação à saída
            da operação backpropagada pelo filho.

        - args: variaveis de entrada da operacao (pais)
            como tensores.

        - O nome dos tensores de gradiente devem ter o
            nome da operacao seguido de '_grad'.
        """

class Add(Op):
    """Add(a, b): a + b"""
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        a, b = args

        if not isinstance(a, Tensor):
            a = Tensor(a, requires_grad=False)
        
        if not isinstance(b, Tensor):
            b = Tensor(b, requires_grad=False)

        resultado_array = a.numpy() + b.numpy()
        tensor_filho = Tensor(arr=resultado_array, 
                              parents=[a, b],    
                              operation=self,     
                              name=NameManager.new('add'),
                              requires_grad=any(p.requires_grad for p in [a, b]))
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        # Como a derivada de a + b em relação a a e b é 1, a derivada parcial em relação a cada pai é igual a 1
        a, b = args
        back_grad_arr = back_grad.numpy()

        grad_a_arr = back_grad_arr
        if a.shape != grad_a_arr.shape:
            axis_sum = tuple(i for i, dim in enumerate(a.shape) if dim == 1)
            grad_a_arr = np.sum(grad_a_arr, axis=axis_sum, keepdims=True)

        grad_b_arr = back_grad_arr
        if b.shape != grad_b_arr.shape:
            axis_sum = tuple(i for i, dim in enumerate(b.shape) if dim == 1)
            grad_b_arr = np.sum(grad_b_arr, axis=axis_sum, keepdims=True)

        grad_a = Tensor(grad_a_arr, requires_grad=False, name=NameManager.new('add_grad'))
        grad_b = Tensor(grad_b_arr, requires_grad=False, name=NameManager.new('add_grad'))
        
        assert grad_a.shape == a.shape and grad_b.shape == b.shape
        return [grad_a, grad_b]
    
# Instancia a classe. O objeto passa a poder ser usado como uma funcao
add = Add()

class Sub(Op):
    """Sub(a, b): a - b"""
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        a, b = args

        if not isinstance(a, Tensor):
            a = Tensor(a, requires_grad=False)
        
        if not isinstance(b, Tensor):
            b = Tensor(b, requires_grad=False)
            
        resultado_array = a.numpy() - b.numpy()
        tensor_filho = Tensor(arr=resultado_array, 
                              parents=[a, b],    
                              operation=self,     
                              name=NameManager.new('sub'),
                              requires_grad=any(p.requires_grad for p in [a, b]))
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        # assim como na soma, na subtração as derivadas serão 1 e -1
        a, b = args
        back_grad_arr = back_grad.numpy()

        grad_a_arr = back_grad_arr
        if a.shape != grad_a_arr.shape:
            axis_sum = tuple(i for i, dim in enumerate(a.shape) if dim == 1)
            grad_a_arr = np.sum(grad_a_arr, axis=axis_sum, keepdims=True)

        grad_b_arr = -1 * back_grad_arr
        if b.shape != grad_b_arr.shape:
            axis_sum = tuple(i for i, dim in enumerate(b.shape) if dim == 1)
            grad_b_arr = np.sum(grad_b_arr, axis=axis_sum, keepdims=True)

        grad_a = Tensor(grad_a_arr, requires_grad=False, name=NameManager.new('sub_grad'))
        grad_b = Tensor(grad_b_arr, requires_grad=False, name=NameManager.new('sub_grad'))
        
        assert grad_a.shape == a.shape and grad_b.shape == b.shape
        return [grad_a, grad_b]


# Instancia a classe. O objeto passa a poder ser usado como uma funcao
sub = Sub()

class Prod(Op):
    """Prod(a, b): produto ponto a ponto de a e b ou produto escalar-tensor"""
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        a, b = args

        if not isinstance(a, Tensor):
            a = Tensor(a, requires_grad=False)
        
        if not isinstance(b, Tensor):
            b = Tensor(b, requires_grad=False)
            
        resultado_array = a.numpy() * b.numpy()
        tensor_filho = Tensor(arr=resultado_array, 
                              parents=[a, b],    
                              operation=self,     
                              name=NameManager.new('prod'),
                              requires_grad=any(p.requires_grad for p in [a, b]))
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        # a derivada em relação a um será o outro, logo
        a, b = args
        back_grad_arr = back_grad.numpy()

        grad_a_arr = b.numpy() * back_grad_arr
        if a.shape != grad_a_arr.shape:
            axis_sum = tuple(i for i, dim in enumerate(a.shape) if dim == 1)
            grad_a_arr = np.sum(grad_a_arr, axis=axis_sum, keepdims=True)
            
        grad_b_arr = a.numpy() * back_grad_arr
        if b.shape != grad_b_arr.shape:
            axis_sum = tuple(i for i, dim in enumerate(b.shape) if dim == 1)
            grad_b_arr = np.sum(grad_b_arr, axis=axis_sum, keepdims=True)

        grad_a = Tensor(grad_a_arr, requires_grad=False, name=NameManager.new('prod_grad'))
        grad_b = Tensor(grad_b_arr, requires_grad=False, name=NameManager.new('prod_grad'))
        
        assert grad_a.shape == a.shape and grad_b.shape == b.shape
        return [grad_a, grad_b]

# Instancia a classe. O objeto passa a poder ser usado como uma funcao
prod = Prod()

class Sin(Op):
    """seno element-wise"""
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        a = args[0]
        resultado_array = np.sin(a.numpy())
        tensor_filho = Tensor(arr=resultado_array,
                              parents=list(args),
                              operation=self,
                              name=NameManager.new('sin'),
                              requires_grad=a.requires_grad)
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        a = args[0]
        resultado_array = back_grad.numpy() * np.cos(a.numpy())
        grad_a = Tensor(resultado_array,
                        requires_grad=False,
                        name=NameManager.new('sin_grad'))
        assert grad_a.shape == a.shape
        return [grad_a]
    
# Instancia a classe. O objeto passa a poder ser usado como uma funcao
sin = Sin()

class Cos(Op):
    """cosseno element-wise"""
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        a = args[0]
        resultado_array = np.cos(a.numpy())
        tensor_filho = Tensor(arr=resultado_array,
                              parents=list(args),
                              operation=self,
                              name=NameManager.new('cos'),
                              requires_grad=a.requires_grad)
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        a = args[0]
        resultado_array = back_grad.numpy() * (-1 * np.sin(a.numpy()))
        grad_a = Tensor(resultado_array,
                        requires_grad=False,
                        name=NameManager.new('cos_grad'))
        assert grad_a.shape == a.shape
        return [grad_a]
    
# Instancia a classe. O objeto passa a poder ser usado como uma funcao
cos = Cos()

class Sum(Op):
    """Retorna a soma dos elementos do tensor"""
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        a = args[0]
        resultado_array = np.sum(a.numpy())
        tensor_filho = Tensor(arr=resultado_array,
                              parents=list(args),
                              operation=self,
                              name=NameManager.new('sum'),
                              requires_grad=a.requires_grad)
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        a = args[0]
        grad_array = np.ones_like(a.numpy()) * back_grad.numpy()
        grad_a = Tensor(grad_array,
                        requires_grad=False,
                        name=NameManager.new('sum_grad'))
        assert grad_a.shape == a.shape
        return [grad_a]

# Instancia a classe. O objeto passa a poder ser usado como uma funcao
# chamar de my_sum porque python ja possui uma funcao sum
my_sum = Sum()

class Mean(Op):
    """Retorna a média dos elementos do tensor"""
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        a = args[0]
        resultado_array = np.mean(a.numpy())
        tensor_filho = Tensor(arr=resultado_array,
                              parents=list(args),
                              operation=self,
                              name=NameManager.new('mean'),
                              requires_grad=a.requires_grad)
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        a = args[0]
        N = a.numpy().size
        grad_array = np.ones_like(a.numpy()) * (back_grad.numpy() / N)
        grad_a = Tensor(grad_array,
                        requires_grad=False,
                        name=NameManager.new('mean_grad'))
        assert grad_a.shape == a.shape
        return [grad_a]

# Instancia a classe. O objeto passa a poder ser usado como uma funcao
mean = Mean()

class Square(Op):
    """Eleva cada elemento ao quadrado"""
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        a = args[0]
        resultado_array = np.square(a.numpy())
        tensor_filho = Tensor(arr=resultado_array,
                              parents=list(args),
                              operation=self,
                              name=NameManager.new('square'),
                              requires_grad=a.requires_grad)
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        a = args[0]
        resultado_array = 2 * a.numpy() * back_grad.numpy()
        grad_a = Tensor(resultado_array,
                        requires_grad=False,
                        name=NameManager.new('square_grad'))
        assert grad_a.shape == a.shape
        return [grad_a]

# Instancia a classe. O objeto passa a poder ser usado como uma funcao
square = Square()

class MatMul(Op):
    """MatMul(A, B): multiplicação de matrizes

    C = A @ B
    de/dA = de/dc @ B^T
    de/dB = A^T @ de/dc

    """

    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        A, B = args

        if not isinstance(A, Tensor):
            A = Tensor(A, requires_grad=False)
        
        if not isinstance(B, Tensor):
            B = Tensor(B, requires_grad=False)

        assert A.shape[1] == B.shape[0], f"Shapes incompatíveis para MatMul: {A.shape} vs {B.shape}"
        resultado_array = A.numpy() @ B.numpy()
        tensor_filho = Tensor(arr=resultado_array,
                              parents=[A, B],
                              operation=self,
                              name=NameManager.new('matmul'),
                              requires_grad=A.requires_grad or B.requires_grad)
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        A, B = args
        # Derivada em relação a A: de/dA = de/dc @ B^T
        grad_A = Tensor(back_grad.numpy() @ B.numpy().T,
                        requires_grad=False,
                        name=NameManager.new('matmul_grad_A'))
        # Derivada em relação a B: de/dB = A^T @ de/dc
        grad_B = Tensor(A.numpy().T @ back_grad.numpy(),
                        requires_grad=False,
                        name=NameManager.new('matmul_grad_B'))
        assert grad_A.shape == A.shape and grad_B.shape == B.shape
        return [grad_A, grad_B]

# Instancia a classe. O objeto passa a poder ser usado como uma funcao
matmul = MatMul()

class Exp(Op):
    """Exponenciação element-wise"""
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        a = args[0]
        resultado_array = np.exp(a.numpy())
        tensor_filho = Tensor(arr=resultado_array,
                              parents=list(args),
                              operation=self,
                              name=NameManager.new('exp'),
                              requires_grad=a.requires_grad)
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        a = args[0]
        resultado_array = back_grad.numpy() * np.exp(a.numpy())
        grad_a = Tensor(resultado_array,
                        requires_grad=False,
                        name=NameManager.new('exp_grad'))
        assert grad_a.shape == a.shape
        return [grad_a]

# Instancia a classe. O objeto passa a poder ser usado como uma funcao
exp = Exp()

class ReLU(Op):
    """ReLU element-wise"""
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        a = args[0]
        resultado_array = np.maximum(0, a.numpy())
        tensor_filho = Tensor(arr=resultado_array,
                              parents=list(args),
                              operation=self,
                              name=NameManager.new('relu'),
                              requires_grad=a.requires_grad)
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        a = args[0]
        # A derivada é 1 onde a > 0 e 0 onde a <= 0
        relu_grad = np.where(a.numpy() > 0, 1, 0)
        resultado_array = relu_grad * back_grad.numpy()
        grad_a = Tensor(resultado_array,
                        requires_grad=False,
                        name=NameManager.new('relu_grad'))
        assert grad_a.shape == a.shape
        return [grad_a]
    
# Instancia a classe. O objeto passa a poder ser usado como uma funcao
relu = ReLU()

class Sigmoid(Op):
    """Sigmoid element-wise"""
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        a = args[0]
        resultado_array = 1 / (1 + np.exp(-a.numpy()))
        tensor_filho = Tensor(arr=resultado_array,
                              parents=list(args),
                              operation=self,
                              name=NameManager.new('sigmoid'),
                              requires_grad=a.requires_grad)
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        a = args[0]
        sigmoid_value = 1 / (1 + np.exp(-a.numpy()))
        resultado_array = sigmoid_value * (1 - sigmoid_value) * back_grad.numpy()
        grad_a = Tensor(resultado_array,
                        requires_grad=False,
                        name=NameManager.new('sigmoid_grad'))
        assert grad_a.shape == a.shape
        return [grad_a]

# Instancia a classe. O objeto passa a poder ser usado como uma funcao
sigmoid = Sigmoid()

class Tanh(Op):
    """Tanh element-wise"""
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        a = args[0]
        resultado_array = np.tanh(a.numpy())
        tensor_filho = Tensor(arr=resultado_array,
                              parents=list(args),
                              operation=self,
                              name=NameManager.new('tanh'),
                              requires_grad=a.requires_grad)
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        a = args[0]
        tanh_value = np.tanh(a.numpy())
        resultado_array = (1 - tanh_value ** 2) * back_grad.numpy()
        grad_a = Tensor(resultado_array,
                        requires_grad=False,
                        name=NameManager.new('tanh_grad'))
        assert grad_a.shape == a.shape
        return [grad_a]

# Instancia a classe. O objeto passa a poder ser usado como uma funcao
tanh = Tanh()

class Softmax(Op):
    """Softmax de um array de valores. Lembre-se que cada elemento do array influencia o resultado da função para todos os demais elementos."""
    def __call__(self, *args, **kwargs) -> Tensor:
        """Realiza a operação usando os argumentos dados em args"""
        a = args[0]
        a_numpy = a.numpy()
        
        max_vals = np.max(a_numpy, axis=0, keepdims=True)
        a_shifted = a_numpy - max_vals # para manter a estabilidade numérica
        exps = np.exp(a_shifted)
        sum_of_exps = np.sum(exps, axis=0, keepdims=True)
        resultado_array = exps / sum_of_exps
        
        # salvar a saída para uso eficiente no prox metodo
        self.softmax_output = resultado_array
        
        tensor_filho = Tensor(arr=resultado_array,
                              parents=list(args),
                              operation=self,
                              name=NameManager.new('softmax'),
                              requires_grad=a.requires_grad)
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        a = args[0]
        s = self.softmax_output
        grad_s = back_grad.numpy()
        
        dot_product = np.sum(grad_s * s, axis=0, keepdims=True)
        grad_array = s * (grad_s - dot_product)

        grad_a = Tensor(grad_array,
                        requires_grad=False,
                        name=NameManager.new('softmax_grad'))
        assert grad_a.shape == a.shape
        return [grad_a]

# Instancia a classe. O objeto passa a poder ser usado como uma funcao
softmax = Softmax()

class Pow(Op):
    """Eleva um tensor a uma potência escalar."""
    def __call__(self, *args, **kwargs) -> Tensor:
        base_tensor, exponent = args
        self._exponent = exponent # Guardamos o expoente para o gradiente

        # Garante que a base seja um Tensor
        if not isinstance(base_tensor, Tensor):
            base_tensor = Tensor(base_tensor, requires_grad=False)

        resultado_array = base_tensor.numpy() ** self._exponent
        tensor_filho = Tensor(arr=resultado_array,
                              parents=[base_tensor],
                              operation=self,
                              name=NameManager.new('pow'),
                              requires_grad=base_tensor.requires_grad)
        return tensor_filho

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a derivada em relação à base."""
        base_tensor = args[0]
        # Derivada: p * x^(p-1) * gradiente_do_filho
        grad_array = self._exponent * (base_tensor.numpy() ** (self._exponent - 1))
        resultado_array = grad_array * back_grad.numpy()

        grad_a = Tensor(resultado_array,
                        requires_grad=False,
                        name=NameManager.new('pow_grad'))
        assert grad_a.shape == base_tensor.shape
        return [grad_a]

# Instancie a classe no final do arquivo ops.py
power = Pow()