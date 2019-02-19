import numpy
import scipy.special
import matplotlib.pyplot

#三层网络
class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #输入层到隐藏层的矩阵
        self.wih = numpy.random.normal(0.0,pow(self.hnodes, -0.5),
                                       (self.hnodes,self.inodes))

        #隐藏层到输出层的矩阵
        self.who = numpy.random.normal(0.0,pow(self.onodes, -0.5),
                                       (self.onodes,self.hnodes))
        #学习率
        self.lr = learningrate
        #激活函数
        self.activation = lambda x:scipy.special.expit(x)
        pass

    def train(self, input_list, targets_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation(final_inputs)

        #反向传播的误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),
                                        numpy.transpose(hidden_outputs))

        self.wih += self.lr * numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation(final_inputs)

        return final_outputs

n = neuralNetwork(6,6,6,0.3)
for i in range(5000):
    n.train([0.1,0.2,0.3,0.4,0.5,0.6],[0.6,0.5,0.4,0.3,0.2,0.1])
    n.train([0.6,0.5,0.4,0.3,0.2,0.1],[0.1,0.2,0.3,0.4,0.5,0.6],)
    if i%500==0:
        print(n.query([0.1,0.2,0.3,0.4,0.5,0.6]))
        print(n.query([0.6,0.5,0.4,0.3,0.2,0.1]))
        print(" ")
