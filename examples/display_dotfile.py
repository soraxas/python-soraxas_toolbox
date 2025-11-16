import pydot

import soraxas_toolbox as st

dot_string = """graph my_graph {
    bgcolor="cyan";
    a [label="Foo"];
    b [shape=circle];
    a -- b -- c [color=blue];
}"""

graphs = pydot.graph_from_dot_data(dot_string)
assert graphs is not None

st.image.display(*graphs)
