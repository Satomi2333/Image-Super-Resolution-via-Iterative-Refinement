def make_product(l: list):
    '''
    make a product of input.
    input: [(func1, [values1]), (func2, [values2]), ...]
             func is an lambda function like "lambda x: x['model']['diffusion']"
             [values] is a list of hyper-parameters like [1, 2, 5]
    output: [(func1, values1[0]), (func2, values2[0]), ..]
            ...
            [(func1, values1[0]), (func2, values2[n-1]), ..]
            [(func1, values1[1]), (func2, values2[0]), ..]
            ...
            [(func1, values1[1]), (func2, values2[n-1]), ..]
            [(func1, values1[2]), (func2, values2[0]), ..]
            ...
            [(func1, values1[n-1]), (func2, values2[0]), ..]
            ...
            [(func1, values1[n-1]), (func2, values2[n-1]), ..]

    '''
    if not l:
        yield []
    else:
        for element in l[0][1]:
            for rest in make_product(l[1:]):
                yield [(l[0][0], element)] + rest
