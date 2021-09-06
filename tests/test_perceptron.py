# testing for perceptron -- if you are looking at this file as it is, it is not usable--I moved this here from the perceptron file to clean it up
# this file will be updated later

# 2D
bias = 5
data = [[r.randint(-50, 50), r.randint(-50, 50)] for _ in range(400)]
data = [[1] + datum if datum[0] + bias > datum[1] else [-1] + datum for datum in data]

accuracy_test_data = [[r.randint(-50, 50), r.randint(-50, 50)] for _ in range(100)]
accuracy_test_data = [[1] + datum if datum[0] + bias > datum[1] else [-1] + datum for datum in accuracy_test_data]

# 3D

p = Perceptron()
p.train(data)
print('accuracy with original dataset: ', p.test_accuracy(data))
print('accuracy for accuracy_test_data', p.test_accuracy(accuracy_test_data))