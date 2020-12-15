import os
import subprocess

def _dot_var(v, verbose = False):
    
    name = '' if v.name is None else v.name
    color = 'orange' if v.name is not None else 'khaki'
    
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += f'{str(v.shape)} {str(v.dtype)}'
        
    return f'{id(v)} [label="{name}", color={color}, style=filled]\n'

def _dot_func(f):
    
    txt = f'{id(f)} [label="{f.__class__.__name__}", color=lightblue, style=filled, shape=box]\n'
    
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y())) # y is wearref
        
    return txt
        
def get_dot_graph(output, verbose=True):

    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            
            if x.creator is not None:
                add_func(x.creator)
    
    return 'digraph g {\n' + txt + '}' 

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)
    
    # save dot data to file
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')
    
    with open(graph_path, 'w') as f:
        f.write(dot_graph)
    
    # call dot command
    extension = os.path.splitext(to_file)[1][1:] # 확장자(png, pdf 등)
    cmd = f'dot {graph_path} -T {extension} -o {to_file}'
    subprocess.run(cmd, shell=True)

    # Return the image as a Jupyter Image object, to be displayed in-line.
    try:
        from IPython import display
        return display.Image(filename=to_file)
    except:
        pass