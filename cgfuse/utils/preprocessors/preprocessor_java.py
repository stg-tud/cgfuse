from . import Preprocessor


class PreprocessorJava(Preprocessor):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.language = "java"
        self.assignment = ["assignment_expression"]
        self.def_statement = ["variable_declarator"]
        self.increment_statement = ["update_expression"]
        self.if_statement = ["if_statement", "else"]
        self.for_statement = ["for_statement"]
        self.enhanced_for_statement = ["enhanced_for_statement"]
        self.while_statement = ["while_statement"]
        self.do_first_statement = []

    def extract_dfg(self, root_node, index_to_code, states, skip_first=True):
        states = states.copy()
        nonescape_nodes = self.filter_instr_nodes(root_node)
        if (
            len(nonescape_nodes) == 0 # or root_node.type == "string"
        ) and "comment" not in root_node.type:
            idx, code = self.get_idx_code(root_node, index_to_code, skip_first)
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
                indexs = self.tree_to_variable_index(
                    name, index_to_code, skip_first
                )
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
            left_nodes = root_node.child_by_field_name("left")
            right_nodes = root_node.child_by_field_name("right")
            DFG = []
            temp, states = self.extract_dfg(right_nodes, index_to_code, states, skip_first)
            DFG += temp
            name_indexs = self.tree_to_variable_index(
                left_nodes, index_to_code, skip_first
            )
            value_indexs = self.tree_to_variable_index(
                right_nodes, index_to_code, skip_first
            )
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, "computedFrom", [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(DFG, key=lambda x: x[1]), states
        elif root_node.type in self.increment_statement:
            DFG = []
            indexs = self.tree_to_variable_index(
                root_node, index_to_code, skip_first
            )
            for index1 in indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, "computedFrom", [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(DFG, key=lambda x: x[1]), states
        elif root_node.type in self.if_statement:
            DFG = []
            current_states = states.copy()
            others_states = []
            flag = False
            tag = False
            if "else" in root_node.type:
                tag = True
            for child in root_node.children:
                if "else" in child.type:
                    tag = True
                if child.type not in self.if_statement and flag is False:
                    temp, current_states = self.extract_dfg(
                        child, index_to_code, current_states, skip_first
                    )
                    DFG += temp
                else:
                    flag = True
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
            for child in root_node.children:
                temp, states = self.extract_dfg(child, index_to_code, states, skip_first)
                DFG += temp
            flag = False
            for child in root_node.children:
                if flag:
                    temp, states = self.extract_dfg(child, index_to_code, states, skip_first)
                    DFG += temp
                elif child.type == "local_variable_declaration":
                    flag = True
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
        elif root_node.type in self.enhanced_for_statement:
            name = root_node.child_by_field_name("name")
            value = root_node.child_by_field_name("value")
            body = root_node.child_by_field_name("body")
            DFG = []
            for i in range(2):
                temp, states = self.extract_dfg(value, index_to_code, states, skip_first)
                DFG += temp
                name_indexs = self.tree_to_variable_index(
                    name, index_to_code, skip_first
                )
                value_indexs = self.tree_to_variable_index(
                    value, index_to_code, skip_first
                )
                for index1 in name_indexs:
                    idx1, code1 = index_to_code[index1]
                    for index2 in value_indexs:
                        idx2, code2 = index_to_code[index2]
                        DFG.append((code1, idx1, "computedFrom", [code2], [idx2]))
                    states[code1] = [idx1]
                temp, states = self.extract_dfg(body, index_to_code, states, skip_first)
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
