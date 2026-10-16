from . import Preprocessor


class PreprocessorPython(Preprocessor):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.language = "python"
        self.assignment = ["assignment", "augmented_assignment", "for_in_clause"]
        self.if_statement = ["if_statement"]
        self.for_statement = ["for_statement"]
        self.while_statement = ["while_statement"]
        self.do_first_statement = ["for_in_clause"]
        self.def_statement = ["default_parameter"]

    def extract_dfg(self, root_node, index_to_code, states, skip_first=False):
        states = states.copy()
        nonescape_nodes = self.filter_instr_nodes(root_node)
        if (
            len(nonescape_nodes) == 0 # or root_node.type == "string"
        ) and "comment" not in root_node.type:
            idx, code = self.get_idx_code(root_node, index_to_code)
            if root_node.type == code:
                return [], states
            elif code in states:
                return [(code, idx, "comesFrom", [code], states[code].copy())], states
            else:
                if root_node.type == "identifier":
                    states[code] = [idx]
                return [(code, idx, "comesFrom", [], [])], states
        elif root_node.type in self.def_statement:
            name = root_node.child_by_field_name("name")
            value = root_node.child_by_field_name("value")
            DFG = []
            if value is None:
                indexs = self.tree_to_variable_index(name, index_to_code, skip_first)
                for index in indexs:
                    idx, code = index_to_code[index]
                    DFG.append((code, idx, "comesFrom", [], []))
                    states[code] = [idx]
                return sorted(DFG, key=lambda x: x[1]), states
            else:
                name_indexs = self.tree_to_variable_index(
                    name, index_to_code, skip_first
                )
                value_indexs = self.tree_to_variable_index(
                    value, index_to_code, skip_first
                )
                temp, states = self.extract_dfg(value, index_to_code, states, skip_first)
                DFG += temp
                for index1 in name_indexs:
                    idx1, code1 = index_to_code[index1]
                    for index2 in value_indexs:
                        idx2, code2 = index_to_code[index2]
                        DFG.append((code1, idx1, "comesFrom", [code2], [idx2]))
                    states[code1] = [idx1]
                return sorted(DFG, key=lambda x: x[1]), states
        elif root_node.type in self.assignment:
            if root_node.type == "for_in_clause":
                right_nodes = [root_node.children[-1]]
                left_nodes = [root_node.child_by_field_name("left")]
            else:
                if root_node.child_by_field_name("right") is None:
                    return [], states
                left_nodes = [
                    x
                    for x in root_node.child_by_field_name("left").children
                    if x.type != ","
                ]
                right_nodes = [
                    x
                    for x in root_node.child_by_field_name("right").children
                    if x.type != ","
                ]
                if len(right_nodes) != len(left_nodes):
                    left_nodes = [root_node.child_by_field_name("left")]
                    right_nodes = [root_node.child_by_field_name("right")]
                if len(left_nodes) == 0:
                    left_nodes = [root_node.child_by_field_name("left")]
                if len(right_nodes) == 0:
                    right_nodes = [root_node.child_by_field_name("right")]
            DFG = []
            for node in right_nodes:
                temp, states = self.extract_dfg(node, index_to_code, states, skip_first)
                DFG += temp

            for left_node, right_node in zip(left_nodes, right_nodes):
                left_tokens_index = self.tree_to_variable_index(
                    left_node, index_to_code, skip_first
                )
                right_tokens_index = self.tree_to_variable_index(
                    right_node, index_to_code, skip_first
                )
                temp = []
                for token1_index in left_tokens_index:
                    idx1, code1 = index_to_code[token1_index]
                    temp.append(
                        (
                            code1,
                            idx1,
                            "computedFrom",
                            [index_to_code[x][1] for x in right_tokens_index],
                            [index_to_code[x][0] for x in right_tokens_index],
                        )
                    )
                    states[code1] = [idx1]
                DFG += temp
            return sorted(DFG, key=lambda x: x[1]), states
        elif root_node.type in self.if_statement:
            DFG = []
            current_states = states.copy()
            others_states = []
            tag = False
            if "else" in root_node.type:
                tag = True
            for child in root_node.children:
                if "else" in child.type:
                    tag = True
                if child.type not in ["elif_clause", "else_clause"]:
                    temp, current_states = self.extract_dfg(
                        child, index_to_code, current_states, skip_first
                    )
                    DFG += temp
                else:
                    temp, new_states = self.extract_dfg(child, index_to_code, states, skip_first)
                    DFG += temp
                    others_states.append(new_states)
            others_states.append(current_states)
            if tag is False:
                others_states.append(states)
            new_states = {}
            for dic in others_states:
                for key in dic:
                    if key not in new_states:
                        new_states[key] = dic[key].copy()
                    else:
                        new_states[key] += dic[key]
            for key in new_states:
                new_states[key] = sorted(list(set(new_states[key])))
            return sorted(DFG, key=lambda x: x[1]), new_states
        elif root_node.type in self.for_statement:
            DFG = []
            for i in range(2):
                right_nodes = [
                    x
                    for x in root_node.child_by_field_name("right").children
                    if x.type != ","
                ]
                left_nodes = [
                    x
                    for x in root_node.child_by_field_name("left").children
                    if x.type != ","
                ]
                if len(right_nodes) != len(left_nodes):
                    left_nodes = [root_node.child_by_field_name("left")]
                    right_nodes = [root_node.child_by_field_name("right")]
                if len(left_nodes) == 0:
                    left_nodes = [root_node.child_by_field_name("left")]
                if len(right_nodes) == 0:
                    right_nodes = [root_node.child_by_field_name("right")]
                for node in right_nodes:
                    temp, states = self.extract_dfg(node, index_to_code, states, skip_first)
                    DFG += temp
                for left_node, right_node in zip(left_nodes, right_nodes):
                    left_tokens_index = self.tree_to_variable_index(
                        left_node, index_to_code, skip_first
                    )
                    right_tokens_index = self.tree_to_variable_index(
                        right_node, index_to_code, skip_first
                    )
                    temp = []
                    for token1_index in left_tokens_index:
                        idx1, code1 = index_to_code[token1_index]
                        temp.append(
                            (
                                code1,
                                idx1,
                                "computedFrom",
                                [index_to_code[x][1] for x in right_tokens_index],
                                [index_to_code[x][0] for x in right_tokens_index],
                            )
                        )
                        states[code1] = [idx1]
                    DFG += temp
                if root_node.children[-1].type == "block":
                    temp, states = self.extract_dfg(
                        root_node.children[-1], index_to_code, states, skip_first
                    )
                    DFG += temp
            dic = {}
            for x in DFG:
                if (x[0], x[1], x[2]) not in dic:
                    dic[(x[0], x[1], x[2])] = [x[3], x[4]]
                else:
                    dic[(x[0], x[1], x[2])][0] = list(
                        set(dic[(x[0], x[1], x[2])][0] + x[3])
                    )
                    dic[(x[0], x[1], x[2])][1] = sorted(
                        list(set(dic[(x[0], x[1], x[2])][1] + x[4]))
                    )
            DFG = [
                (x[0], x[1], x[2], y[0], y[1])
                for x, y in sorted(dic.items(), key=lambda t: t[0][1])
            ]
            return sorted(DFG, key=lambda x: x[1]), states
        elif root_node.type in self.while_statement:
            DFG = []
            for i in range(2):
                for child in root_node.children:
                    temp, states = self.extract_dfg(child, index_to_code, states, skip_first)
                    DFG += temp
            dic = {}
            for x in DFG:
                if (x[0], x[1], x[2]) not in dic:
                    dic[(x[0], x[1], x[2])] = [x[3], x[4]]
                else:
                    dic[(x[0], x[1], x[2])][0] = list(
                        set(dic[(x[0], x[1], x[2])][0] + x[3])
                    )
                    dic[(x[0], x[1], x[2])][1] = sorted(
                        list(set(dic[(x[0], x[1], x[2])][1] + x[4]))
                    )
            DFG = [
                (x[0], x[1], x[2], y[0], y[1])
                for x, y in sorted(dic.items(), key=lambda t: t[0][1])
            ]
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            DFG = []
            for child in root_node.children:
                if child.type in self.do_first_statement:
                    temp, states = self.extract_dfg(child, index_to_code, states, skip_first)
                    DFG += temp
            for child in root_node.children:
                if child.type not in self.do_first_statement:
                    temp, states = self.extract_dfg(child, index_to_code, states, skip_first)
                    DFG += temp

            return sorted(DFG, key=lambda x: x[1]), states
