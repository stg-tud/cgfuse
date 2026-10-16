from . import Preprocessor
import networkx as nx

def add_indention_markers(graph, start_ind=0, root_node="AST#module", print_code=True, debug=False):
    # Implemented for python, tested on evalplus
    # Get the internal ast graph
    ast_graph = nx.subgraph_view(
        graph.copy(),
        filter_edge=lambda n1, n2, ek: graph[n1][n2][ek]["y"] in [0],
    )

    root_attr = ast_graph.nodes[root_node]
    root_attr["ind"] = start_ind
    root_attr["nl"] = False
    # AST#Module assumed to be root always
    for parent, children in nx.bfs_successors(ast_graph, root_node):
        parent_attr = ast_graph.nodes[parent]
        if debug:
            print(parent, children)
            if "first_token" in parent_attr:
                del parent_attr["first_token"]
            if "start_point" in parent_attr:
                del parent_attr["start_point"]
            if "end_point" in parent_attr:
                del parent_attr["end_point"]
            print(parent_attr)

        if "nl" not in parent_attr:
            parent_attr["nl"] = False
        if "ind" not in parent_attr:
            parent_attr["ind"] = 0

        ast_graph.nodes[children[0]]["ind"] = parent_attr["ind"]
        ast_graph.nodes[children[-1]]["nl"] = parent_attr["nl"]

        for child in children:
            attr = ast_graph.nodes[child]
            if debug:
                if "first_token" in attr:
                    del attr["first_token"]
                if "start_point" in attr:
                    del attr["start_point"]
                if "end_point" in attr:
                    del attr["end_point"]

            if "nl" not in attr:
                ast_graph.nodes[child]["nl"] = False
            if "ind" not in attr:
                ast_graph.nodes[child]["ind"] = 0

            # if "AST#module" in parent:
            #     ast_graph.nodes[child]["ind"] = parent_attr["ind"]
            #     ast_graph.nodes[child]["nl"] = False
            if "AST#block" in parent:
                ast_graph.nodes[child]["ind"] = parent_attr["ind"]

        if "AST#if_statement" in parent:
            for child in children:
                if ":" in child:
                    # ast_graph.nodes[child]["ind"] = parent_attr["ind"]
                    ast_graph.nodes[child]["nl"] = True
                if "block" in child:
                    ast_graph.nodes[child]["ind"] = parent_attr["ind"] + 1
                    ast_graph.nodes[child]["nl"] = True
                elif "AST#elif_clause" in child:
                    ast_graph.nodes[child]["ind"] = parent_attr["ind"]
                    ast_graph.nodes[child]["nl"] = True
                elif "AST#else_clause" in child:
                    ast_graph.nodes[child]["ind"] = parent_attr["ind"]
                    ast_graph.nodes[child]["nl"] = True
        elif "AST#elif_clause" in parent:
            ast_graph.nodes[children[-1]]["ind"] = parent_attr["ind"] + 1
            ast_graph.nodes[children[-2]]["nl"] = True
        elif "AST#else_clause" in parent:
            ast_graph.nodes[children[-1]]["ind"] = parent_attr["ind"] + 1
            ast_graph.nodes[children[-2]]["nl"] = True
        elif "AST#for_statement" in parent:
            # ' for##00', ' _##00', ' in##00', 'AST#call##02', ':##08', 'AST#block##07', 'AST#else_clause'
            for child in children:
                if ":" in child:
                    # ast_graph.nodes[child]["ind"] = parent_attr["ind"]
                    ast_graph.nodes[child]["nl"] = True
                if "block" in child:
                    ast_graph.nodes[child]["ind"] = parent_attr["ind"] + 1
                    ast_graph.nodes[child]["nl"] = True
                elif "AST#else_clause" in child:
                    ast_graph.nodes[child]["ind"] = parent_attr["ind"] + 1
                    ast_graph.nodes[child]["nl"] = True
        elif "AST#while_statement" in parent:
            # :
            ast_graph.nodes[children[-1]]["ind"] = parent_attr["ind"] + 1
            # block
            ast_graph.nodes[children[-2]]["nl"] = True
        elif "AST#function_definition" in parent:
            # :
            ast_graph.nodes[children[-1]]["ind"] = parent_attr["ind"] + 1
            # block
            ast_graph.nodes[children[-2]]["nl"] = True
        elif "AST#import_statement" in parent:
            ast_graph.nodes[children[-1]]["nl"] = True
        elif "AST#import_from_statement" in parent:
            ast_graph.nodes[children[-1]]["nl"] = True
        elif "AST#return_statement" in parent:
            ast_graph.nodes[children[-1]]["nl"] = True
        elif "AST#expression_statement" in parent:
            ast_graph.nodes[children[-1]]["nl"] = True
        elif "AST#assert_statement" in parent:
            ast_graph.nodes[children[-1]]["nl"] = True
        elif "AST#continue_statement" in parent:
            ast_graph.nodes[children[-1]]["nl"] = True

        if debug:
            for child in children:
                attr = ast_graph.nodes[child]
                print(attr)
    if print_code:
        for node, node_attr in ast_graph.nodes(data=True):
            if node_attr['code_token']:
                if node_attr['ind'] > 0:
                    print(" " * node_attr['ind'], end='')
                print(node_attr["flat_name"], end='')
                if node_attr['nl']:
                    print()

    return ast_graph

def indented_graph(ast_graph, print_code=True, indent=" "):
    """Adds the indentation and newline nodes
    with code order to the graph"""
    indented = nx.MultiGraph(first_token=False)
    ind_idx, nl_idx = 0, 0
    for node, node_attr in ast_graph.nodes(data=True):
        if node_attr['code_token']:
            if node_attr['ind'] > 0:
                for _ in range(node_attr['ind']):
                    cp_attr = node_attr.copy()
                    cp_attr['flat_name'] = indent
                    indented.add_node(f"{ind_idx:04d}", **cp_attr)
                    ind_idx += 1

            indented.add_node(node, **node_attr)

            if node_attr['nl']:
                cp_attr = node_attr.copy()
                cp_attr['flat_name'] = "\n"
                indented.add_node(f"\n{nl_idx}", **cp_attr)
                nl_idx += 1

            continue
        indented.add_node(node, **node_attr)

    for u, v, edge_attr in list(ast_graph.edges(data=True)):
        indented.add_edge(u, v, **edge_attr)

    if print_code:
        for node, node_attr in indented.nodes(data=True):
            if node_attr['code_token']:
                print(node_attr['flat_name'], end='')

    return indented

def graph_tokenize(G, tokenizer, order_req=False, lang="java", startind=0):
    # ? It might be better to iterate over all nodes to create a new graph
    # ? where code nodes are tokenized, this would preserve ordering most likely

    def get_attr(token):
        if tokenizer.vocab.get(token, None):
            return {
                'flat_name': token,
                'order': -1,
                'code_token': True,
                'ast_type': token
            }
        return None

    # Some languages have these for new lines and spaces
    if lang == "python":
        nlnl = get_attr("ĊĊ")
        nl = get_attr("Ċ")
        ind = get_attr("ĠĠĠ")
        sind = get_attr("Ġ")

    new_G = nx.MultiGraph(first_token=False)
    super_nodes = {}
    order_shift = 0
    curind = 0 + startind
    for super_node in list(G.nodes()):
        # skip syntax nodes
        attr = G.nodes[super_node]
        node_tokens = tokenizer.tokenize(attr.get("flat_name"))

        if not attr.get("code_token"):
            # tokenize applied lower case for some reason,
            # we know that the internal values are normalized already!
            t_ids = tokenizer.convert_tokens_to_ids(attr.get("flat_name"))
            if order_req:
                attr["order"] = attr["order"] + order_shift
            attr["t_id"] = t_ids[0] if isinstance(t_ids, list) else t_ids
            new_G.add_node(super_node, **attr)
            # add edges too
            syntax_node = attr.get("flat_name")
            print(syntax_node)
            if syntax_node == "AST#block":
                curind += 1
                continue

        t_ids = tokenizer.convert_tokens_to_ids(node_tokens)

        # Tokenize the code token
        super_nodes[super_node] = []

        # Iterate over all nodes
        for i, (node_token, t_id) in enumerate(zip(node_tokens, t_ids)):
            # Create a copy of node attributes to add
            new_attr = attr.copy()
            # Update flat_name since this is a subtext
            # ! order value is wrong but it is not required at this point, most likely
            new_attr["flat_name"] = node_token
            new_attr["t_id"] = t_id
            if order_req:
                new_attr["order"] = new_attr["order"] + order_shift
            order_shift += 1
            # ! start_point and end_point are also wrong
            # ! but they might be ok to leave wrong
            # Get unique name so Different nodes are not overwritten
            unique_name = Preprocessor.get_unique_name(node_token, new_G)
            # Add the new node with attributes
            new_G.add_node(unique_name, **new_attr)
            super_nodes[super_node].append(unique_name)
        order_shift -= 1

    for u, v, edge_attr in list(G.edges(data=True)):
        u = super_nodes[u] if u in super_nodes else [u]
        v = super_nodes[v] if v in super_nodes else [v]
        for u_alt in u:
            for v_alt in v:
                new_G.add_edge(u_alt, v_alt, **edge_attr)

    return new_G