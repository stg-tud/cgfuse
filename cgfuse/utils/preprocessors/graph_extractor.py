import re
import tokenize
from io import StringIO


class GraphExtractor:
    # remove comments, tokenize code and extract dataflow
    @staticmethod
    def extract_graph(code, parser, fn_extract_dfg, lang: str, dataset_name: str):
        # remove comments
        try:
            if lang == "python":
                code = GraphExtractor._remove_comments_and_docstrings(code, lang)
        except:
            pass
        # obtain dataflow
        try:
            should_apply_java_wrapper = (
                "concode" in dataset_name.lower()
                or "codesearchnet" in dataset_name.lower()
            ) and lang == "java"
            raw_code = ""

            if should_apply_java_wrapper:
                root_node, code = GraphExtractor.java_class_wrapper(code, parser)
                code_text = root_node.parent.parent.parent.text
            elif lang == "php":
                code = "<?php class foo {\n" + code + "\n}?>"
                tree = parser.parse(code.encode())
                root_node = tree.root_node.children[1].children[2].children[1]
                code_text = tree.root_node.text
            else:
                tree = parser.parse(code.encode())
                root_node = tree.root_node
                code_text = root_node.text

            tokens_index = GraphExtractor._tree_to_token_index(
                root_node, should_apply_java_wrapper, code_text, language=lang
            )

            code_tokens = []
            index_to_code = {}
            for idx, (index, _code) in enumerate(tokens_index):
                code_tokens.append(_code)
                index_to_code[index] = (idx, _code)

            try:
                DFG, _ = fn_extract_dfg(
                    root_node, index_to_code, {}, should_apply_java_wrapper
                )
            except BaseException as err:
                DFG = []
                raise Exception("Could not calculate DFG!") from err
            DFG = sorted(DFG, key=lambda x: x[1])
            indices = set()
            for d in DFG:
                if len(d[-1]) != 0:
                    indices.add(d[1])
                for x in d[-1]:
                    indices.add(x)
            new_DFG = []
            for d in DFG:
                if d[1] in indices:
                    new_DFG.append(d)
            dfg = new_DFG
        except BaseException as err:
            raise Exception("extract_graph failed!") from err
        return root_node, code_tokens, index_to_code, dfg, raw_code

    @staticmethod
    def _remove_comments_and_docstrings(source, lang):
        if lang in ["python"]:
            """
            Returns 'source' minus comments and docstrings.
            """
            io_obj = StringIO(source)
            out = ""
            prev_toktype = tokenize.INDENT
            last_lineno = -1
            last_col = 0
            for tok in tokenize.generate_tokens(io_obj.readline):
                token_type = tok[0]
                token_string = tok[1]
                start_line, start_col = tok[2]
                end_line, end_col = tok[3]
                ltext = tok[4]
                if start_line > last_lineno:
                    last_col = 0
                if start_col > last_col:
                    out += " " * (start_col - last_col)
                # Remove comments:
                if token_type == tokenize.COMMENT:
                    pass
                # This series of conditionals removes docstrings:
                elif token_type == tokenize.STRING:
                    if prev_toktype != tokenize.INDENT:
                        # This is likely a docstring; double-check we're not inside an operator:
                        if prev_toktype != tokenize.NEWLINE:
                            if start_col > 0:
                                out += token_string
                else:
                    out += token_string
                prev_toktype = token_type
                last_col = end_col
                last_lineno = end_line
            temp = []
            for x in out.split("\n"):
                if x.strip() != "":
                    temp.append(x)
            return "\n".join(temp)
        elif lang in ["ruby"]:
            return source
        else:
            def replacer(match):
                s = match.group(0)
                if s.startswith("/"):
                    return " "  # note: a space and not an empty string
                else:
                    return s

            pattern = re.compile(
                r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                re.DOTALL | re.MULTILINE,
            )
            temp = []
            for x in re.sub(pattern, replacer, source).split("\n"):
                if x.strip() != "":
                    temp.append(x)
            return "\n".join(temp)

    @staticmethod
    def _index_to_code_token(index, code):
        start_point = index[0]
        end_point = index[1]
        if start_point[0] == end_point[0]:
            s = code[start_point[0]][start_point[1] : end_point[1]]
        else:
            s = ""
            s += code[start_point[0]][start_point[1] :]
            for i in range(start_point[0] + 1, end_point[0]):
                s += code[i]
            s += code[end_point[0]][: end_point[1]]
        return s

    @classmethod
    def ignored_nodes(cls, language='python'):
        ignored_nodes = ["escape_interpolation", "\\", "comment"]
        if language == "python":
            ignored_nodes.append("escape_sequence")
        if language == "php":
            ignored_nodes.pop(1)  # Keep backslash for now
        return ignored_nodes

    @classmethod
    def filter_instr_nodes(cls, root, language='python'):
        # Copy of it exists in preprocessor. Weird project layout causes
        # circular import therefore copy here.
        return [c for c in root.children if c.type not in cls.ignored_nodes(language)]

    @classmethod
    def _tree_to_token_index(cls, root_node, skip_first=True, code=None, language='python', isstr=False):
        nonescape_nodes = cls.filter_instr_nodes(root_node, language=language)

        if (
            len(nonescape_nodes) == 0 # or root_node.type == "string"
        # ) and root_node.type != "comment":
        ) and "comment" not in root_node.type:
            # return [(root_node.start_point, root_node.end_point)]
            java_skip = (1 if skip_first else 0)
            add_space = False
            if root_node.start_point[1] != 0:
                # prev_char = code[root_node.start_point[0]][root_node.start_point[1]-1:root_node.start_point[1]]
                prev_char = code[root_node.start_byte-1:root_node.start_byte].decode()
                if prev_char in [' ', '\t', '\n']: # Not sure about \t and \n
                    add_space = not isstr or root_node.type in ["string_start", '"']
            return [
                (
                    (
                        (
                            root_node.start_point[0] - java_skip, #row start
                            root_node.start_point[1] - (1 if add_space else 0), #col start
                        ),
                        (
                            root_node.end_point[0] - java_skip, #row end
                            root_node.end_point[1], #col end
                        ),
                    ),
                    # code[root_node.start_byte - (1 if add_space else 0):root_node.end_byte].decode()
                    f"{(' ' if add_space else '')}{code[root_node.start_byte:root_node.end_byte].decode()}"
                )
            ]
        else:
            code_tokens = []
            for i, child in enumerate(root_node.children):
                is_instr = root_node.type in ["string", "string_literal", "interpreted_string_literal", "encapsed_string"]
                is_instr = is_instr and i != 0
                code_tokens += GraphExtractor._tree_to_token_index(child, skip_first, code, language, is_instr or isstr)
            return code_tokens

    @staticmethod
    def java_class_wrapper(code: str, parser):
        """
        tree-sitter assumes to receive Program which should be wrapped in a class.
        When Function is fed, generates error and parse tree is problematic.
        """
        class_wrp_fnc = f"public class HelloWorld {{\n{code}\n}}"
        tree = parser.parse(class_wrp_fnc.encode())
        code_node = tree.root_node.children[0].children[3].children[1]
        return code_node, class_wrp_fnc
