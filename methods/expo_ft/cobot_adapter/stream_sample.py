"""Split native sampling into one index draw and bounded, identical gathers."""
import ast,inspect,textwrap
import numpy as np
_CACHE={}


def functions(buffer):
    kind=type(buffer)
    if kind in _CACHE:return _CACHE[kind]
    original=kind.sample_jax
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    body=tree.body[0].body
    start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='(key, rng)')
    end=next(i for i,n in enumerate(body[start:],start) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='self.rng')+1
    # Execute the original index-selection code verbatim, including success filtering.
    draw=ast.parse(textwrap.dedent(inspect.getsource(original)))
    draw.body[0].name='draw_indices'
    draw.body[0].body=body[:end]+[ast.Return(ast.Name('indices',ast.Load()))]
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(draw),'<native replay index selection>','exec'),namespace)
    gather=ast.parse(textwrap.dedent(inspect.getsource(original)))
    gather.body[0].name='gather_indices'
    gather.body[0].args.args.append(ast.arg(arg='indices'))
    gather.body[0].args.defaults.append(ast.Constant(None))
    gather.body[0].body=body[:start]+body[end:]
    exec(compile(ast.fix_missing_locations(gather),'<native replay gather>','exec'),namespace)
    _CACHE[kind]=(namespace['draw_indices'],namespace['gather_indices'])
    return _CACHE[kind]


class Batch:
    def __init__(self,buffer,total,size):
        draw,self.gather=functions(buffer)
        self.buffer=buffer
        self.indices=np.asarray(draw(buffer,total))
        self.total=total;self.size=size
        if total%size:raise ValueError('Batch must divide into native critic minibatches')
    def __iter__(self):
        for start in range(0,self.total,self.size):
            raw=self.gather(self.buffer,self.size,indices=self.indices[start:start+self.size])
            yield self.buffer._convert_to_openpi_format(raw)
