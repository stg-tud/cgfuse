from functools import partial
from typing import Optional

from omegaconf import DictConfig
from lightning.pytorch.utilities.enums import LightningEnum
from transformers import GenerationConfig
from evaluate import load

from ..utils.metrics import google_bleu, calc_codebleu, ExactMatch


class EmbSource(LightningEnum):
    """Used to denote different train stages in {paper_name}.

      - BOOTSTRAPPING for whole individual model trainings.
      - ADAPTATING for training of teacher models.
      - DISTILLING for infusing information from teacher models.

    provided hard-coded indexes for potential usages with property `i`.
    """

    SELF = "self"
    MODEL = "model"
    LAYER = "layer"

    @property
    def require_merge(self) -> Optional[str]:
        if self in (self.MODEL, self.LAYER):
            return True
        return False


class TaskStage(LightningEnum):
    """Used to denote different train stages in {paper_name}.

      - BOOTSTRAPPING for whole individual model trainings.
      - ADAPTATING for training of teacher models.
      - DISTILLING for infusing information from teacher models.

    provided hard-coded indexes for potential usages with property `i`.
    """

    BOOTSTRAPPING = "bootstrap"
    ADAPTING = "adapt"
    FINETUNING = "finetuning"
    DISTILLING = "distill"

    @property
    def i(self) -> Optional[str]:
        return {
            self.BOOTSTRAPPING: 0,
            self.ADAPTING: 1,
            self.FINETUNING: 2,
            self.DISTILLING: 3,
        }[self]


class Task:

    # Fixed vocabulary of AST node types (all supported languages) added to the tokenizer.
    # Order matters: it fixes the token ids of syntax nodes, so keep it stable.
    syntax_node_types = [
        "AST#null_literal",
        "AST#typed_parameter",
        "AST#list_splat",
        "AST#when",
        "AST#list",
        "AST#unset_statement",
        "AST#abstract_function_declarator",
        "AST#floating_point_type",
        "AST#implicit_array_creation_expression",
        "AST#default_case",
        "AST#and",
        "AST#using_declaration",
        "AST#attribute_specifier",
        "AST#switch_default",
        "AST#field_declaration",
        "AST#order_by_clause",
        "AST#class_interface_clause",
        "AST#int_literal",
        "AST#parameter_pack_expansion",
        "AST#rest_pattern",
        "AST#attribute_declaration",
        "AST#switch_expression_arm",
        "AST#import_from_statement",
        "AST#try_with_resources_statement",
        "AST#class_definition",
        "AST#dimensions_expr",
        "AST#end",
        "AST#elif_clause",
        "AST#yield",
        "AST#hash_bang_line",
        "AST#method_declaration",
        "AST#switch_block_statement_group",
        "AST#query_expression",
        "AST#var_pattern",
        "AST#enum_body_declarations",
        "AST#bracketed_argument_list",
        "AST#define",
        "AST#in",
        "AST#generator_function",
        "AST#assignment_pattern",
        "AST#with_item",
        "AST#expression_switch_statement",
        "AST#unary",
        "AST#operator",
        "AST#jsx_expression",
        "AST#type_switch_statement",
        "AST#catch_formal_parameter",
        "AST#abstract_modifier",
        "AST#element_binding_expression",
        "AST#communication_case",
        "AST#subscript_designator",
        "AST#rational",
        "AST#shorthand_property_identifier_pattern",
        "AST#pair",
        "AST#const_declaration",
        "AST#for_range_loop",
        "AST#real_literal",
        "AST#virtual_specifier",
        "AST#ms_restrict_modifier",
        "AST#name_equals",
        "AST#dictionary_splat_pattern",
        "AST#invocation_expression",
        "AST#interpreted_string_literal",
        "AST#comparison_operator",
        "AST#parameter",
        "AST#tuple_pattern",
        "AST#integral_type",
        "AST#method_invocation",
        "AST#type_parameter_constraint",
        "AST#expression_list",
        "AST#yield_expression",
        "AST#preproc_params",
        "AST#if_modifier",
        "AST#call_expression",
        "AST#translation_unit",
        "AST#augmented_assignment",
        "AST#conditional",
        "AST#parenthesized_expression",
        "AST#type_conversion_expression",
        "AST#send_statement",
        "AST#nil",
        "AST#using_statement",
        "AST#interpolated_verbatim_string_text",
        "AST#get",
        "AST#attribute_argument",
        "AST#dependent_name",
        "AST#interpolated_string_text",
        "AST#var_declaration",
        "AST#for_clause",
        "AST#this_expression",
        "AST#virtual_function_specifier",
        "AST#verbatim_string_literal",
        "AST#nonlocal_statement",
        "AST#switch_body",
        "AST#operator_assignment",
        "AST#import",
        "AST#if_clause",
        "AST#declaration_expression",
        "AST#wildcard",
        "AST#global_declaration",
        "AST#preproc_function_def",
        "AST#constructor_declaration",
        "AST#default_switch_label",
        "AST#array_pattern",
        "AST#template_method",
        "AST#block_argument",
        "AST#template_function",
        "AST#final_modifier",
        "AST#left_assignment_list",
        "AST#retry",
        "AST#when_clause",
        "AST#type_qualifier",
        "AST#qualified_name",
        "AST#abstract_reference_declarator",
        "AST#property_declaration",
        "AST#make_ref_expression",
        "AST#decimal_integer_literal",
        "AST#go_statement",
        "AST#instanceof",
        "AST#continue",
        "AST#hash_splat_argument",
        "AST#marker_annotation",
        "AST#dynamic_variable_name",
        "AST#foreach_statement",
        "AST#reference_declarator",
        "AST#php_tag",
        "AST#scoped_property_access_expression",
        "AST#map_type",
        "AST#splat_argument",
        "AST#nullable_type",
        "AST#bare_symbol",
        "AST#with",
        "AST#ref_expression",
        "AST#scoped_type_identifier",
        "AST#tuple_element",
        "AST#class_declaration",
        "AST#alias",
        "AST#finally_clause",
        "AST#sizeof_expression",
        "AST#type",
        "AST#arrow_function",
        "AST#compilation_unit",
        "AST#operator_cast",
        "AST#ellipsis",
        "AST#meta_property",
        "AST#include",
        "AST#char_literal",
        "AST#subshell",
        "AST#aliased_import",
        "AST#dimensions",
        "AST#symbol_array",
        "AST#throws",
        "AST#interface_type_list",
        "AST#type_arguments",
        "AST#assignment",
        "AST#channel_type",
        "AST#base_class_clause",
        "AST#private_property_identifier",
        "AST#unless_modifier",
        "AST#augmented_assignment_expression",
        "AST#co_return_statement",
        "AST#object_pattern",
        "AST#%w(",
        "AST#field_initializer",
        "AST#class",
        "AST#local_variable_declaration",
        "AST#reference_modifier",
        "AST#explicit_constructor_invocation",
        "AST#else_if_clause",
        "AST#void_keyword",
        "AST#block_parameters",
        "AST#annotation_argument_list",
        "AST#decorator",
        "AST#abstract_array_declarator",
        "AST#template_string",
        "AST#dec_statement",
        "AST#checked_statement",
        "AST#preproc_include",
        "AST#dependent_type",
        "AST#else_directive",
        "AST#trailing_return_type",
        "AST#with_clause",
        "AST#attribute_list",
        "AST#namespace_name",
        "AST#global_statement",
        "AST#dictionary",
        "AST#superclass",
        "AST#heredoc_content",
        "AST#raise_statement",
        "AST#comment",
        "AST#template_type",
        "AST#void",
        "AST#number",
        "AST#lambda",
        "AST#var_spec",
        "AST#rest_assignment",
        "AST#delimited_symbol",
        "AST#class_specifier",
        "AST#break_statement",
        "AST#not",
        "AST#implicit_type",
        "AST#out",
        "AST#preproc_else",
        "AST#inc_statement",
        "AST#while",
        "AST#typed_default_parameter",
        "AST#expression_statement",
        "AST#sequence_expression",
        "AST#preproc_arg",
        "AST#prefix_unary_expression",
        "AST#explicit_function_specifier",
        "AST#print_statement",
        "AST#implicit_length_array_type",
        "AST#heredoc_end",
        "AST#scope_resolution",
        "AST#pointer_expression",
        "AST#this",
        "AST#assignment_expression",
        "AST#regex_flags",
        "AST#chained_string",
        "AST#short_var_declaration",
        "AST#splat_parameter",
        "AST#class_literal",
        "AST#new",
        "AST#for_statement",
        "AST#throw_expression",
        "AST#jsx_text",
        "AST#equals_value_clause",
        "AST#keyword_parameter",
        "AST#elif",
        "AST#namespace_name_as_prefix",
        "AST#unless",
        "AST#base_clause",
        "AST#argument",
        "AST#bare_string",
        "AST#range_clause",
        "AST#dictionary_comprehension",
        "AST#attributed_declarator",
        "AST#import_statement",
        "AST#generic_name",
        "AST#type_parameter_declaration",
        "AST#namespace_identifier",
        "AST#as",
        "AST#list_pattern",
        "AST#preproc_directive",
        "AST#jsx_attribute",
        "AST#optional_parameter_declaration",
        "AST#field_designator",
        "AST#type_parameter",
        "AST#throw_specifier",
        "AST#string_literal",
        "AST#array_rank_specifier",
        "AST#optional_parameter",
        "AST#pass_statement",
        "AST#defer_statement",
        "AST#include_once_expression",
        "AST#annotation",
        "AST#preproc_integer_literal",
        "AST#anonymous_object_creation_expression",
        "AST#interpolation",
        "AST#case_pattern_switch_label",
        "AST#import_prefix",
        "AST#parenthesized_declarator",
        "AST#else_clause",
        "AST#arrow_expression_clause",
        "AST#default_expression",
        "AST#for_in_clause",
        "AST#exceptions",
        "AST#attributed_statement",
        "AST#variadic_argument",
        "AST#union_specifier",
        "AST#comma_expression",
        "AST#boolean_literal",
        "AST#type_argument_list",
        "AST#element_value_array_initializer",
        "AST#lexical_declaration",
        "AST#base_expression",
        "AST#type_constraint",
        "AST#for_each_statement",
        "AST#is_pattern_expression",
        "AST#integer_literal",
        "AST#literal_value",
        "AST#array_creation_expression",
        "AST#rescue_modifier",
        "AST#generic_type",
        "AST#bitfield_clause",
        "AST#ms_declspec_modifier",
        "AST#delete_expression",
        "AST#for",
        "AST#conditional_expression",
        "AST#void_type",
        "AST#shell_command_expression",
        "AST#member_access_expression",
        "AST#raw_string_literal",
        "AST#constant",
        "AST#compound_literal_expression",
        "AST#structured_binding_declarator",
        "AST#template_argument_list",
        "AST#struct_specifier",
        "AST#heredoc",
        "AST#chevron",
        "AST#assert_statement",
        "AST#abstract_pointer_declarator",
        "AST#type_spec",
        "AST#include_expression",
        "AST#enumerator_list",
        "AST#constructor_initializer",
        "AST#until",
        "AST#type_parameters",
        "AST#visibility_modifier",
        "AST#destructured_left_assignment",
        "AST#new_declarator",
        "AST#heredoc_beginning",
        "AST#right_assignment_list",
        "AST#clone_expression",
        "AST#call",
        "AST#else",
        "AST#ensure",
        "AST#undefined",
        "AST#resource",
        "AST#simple_symbol",
        "AST#do",
        "AST#qualified_type",
        "AST#static_variable_declaration",
        "AST#type_definition",
        "AST#sized_type_specifier",
        "AST#identifier",
        "AST#composite_literal",
        "AST#next",
        "AST#new_expression",
        "AST#global",
        "AST#discard",
        "AST#typeof",
        "AST#pointer_declarator",
        "AST#endif_directive",
        "AST#return",
        "AST#parenthesized_statements",
        "AST#singleton_method",
        "AST#finally",
        "AST#attribute",
        "AST#string_fragment",
        "AST#interface_type",
        "AST#enum_declaration",
        "AST#inferred_parameters",
        "AST#func_literal",
        "AST#function_type",
        "AST#method",
        "AST#case_switch_label",
        "AST#list_comprehension",
        "AST#conditional_access_expression",
        "AST#relative_import",
        "AST#hex_integer_literal",
        "AST#alias_declaration",
        "AST#hash",
        "AST#set",
        "AST#default",
        "AST#reference_assignment_expression",
        "AST#co_yield_statement",
        "AST#jsx_element",
        "AST#default_statement",
        "AST#slice_type",
        "AST#switch_label",
        "AST#template_substitution",
        "AST#try_statement",
        "AST#parameter_list",
        "AST#user_defined_literal",
        "AST#regex_pattern",
        "AST#resource_specification",
        "AST#array_element_initializer",
        "AST#function_declarator",
        "AST#endregion_directive",
        "AST#argument_list",
        "AST#delete_statement",
        "AST#anonymous_function_use_clause",
        "AST#accessor_list",
        "AST#jsx_namespace_name",
        "AST#subscript_expression",
        "AST#variadic_parameter",
        "AST#unary_operator",
        "AST#default_method_clause",
        "AST#goto_statement",
        "AST#lock_statement",
        "AST#rune_literal",
        "AST#unary_expression",
        "AST#field_expression",
        "AST#switch_case",
        "AST#then",
        "AST#delete",
        "AST#alias_qualified_name",
        "AST#false",
        "AST#ERROR",
        "AST#decltype",
        "AST#declaration_pattern",
        "AST#colon_block",
        "AST#float_literal",
        "AST#binary_operator",
        "AST#cast_expression",
        "AST#select_statement",
        "AST#method_reference",
        "AST#slice",
        "AST#assignment_statement",
        "AST#co_await_expression",
        "AST#pointer_type",
        "AST#continue_statement",
        "AST#of",
        "AST#preproc_if",
        "AST#class_constant_access_expression",
        "AST#type_list",
        "AST#query_continuation",
        "AST#system_lib_string",
        "AST#assignment_operator",
        "AST#template_parameter_list",
        "AST#yield_statement",
        "AST#string_array",
        "AST#macro_type_specifier",
        "AST#function",
        "AST#string_content",
        "AST#require_once_expression",
        "AST#variable_declarator",
        "AST#select_clause",
        "AST#member_expression",
        "AST#array",
        "AST#qualified_identifier",
        "AST#exponentiation_expression",
        "AST#by_ref",
        "AST#formal_parameter",
        "AST#unsafe_statement",
        "AST#ref_qualifier",
        "AST#super_interfaces",
        "AST#slice_expression",
        "AST#attribute_argument_list",
        "AST#empty_statement",
        "AST#error_directive",
        "AST#scoped_call_expression",
        "AST#default_parameter",
        "AST#literal_suffix",
        "AST#named_label_statement",
        "AST#condition_clause",
        "AST#declare_directive",
        "AST#character",
        "AST#member_binding_expression",
        "AST#property_element",
        "AST#static_assert_declaration",
        "AST#type_of_expression",
        "AST#%i(",
        "AST#template_declaration",
        "AST#boolean_operator",
        "AST#variadic_parameter_declaration",
        "AST#binary_expression",
        "AST#unary_op_expression",
        "AST#tuple",
        "AST#line_directive",
        "AST#do_block",
        "AST#method_spec_list",
        "AST#accessor_declaration",
        "AST#initializer_list",
        "AST#enum_body",
        "AST#throw_statement",
        "AST#arguments",
        "AST#decorated_definition",
        "AST#character_literal",
        "AST#case",
        "AST#elsif",
        "AST#destructured_parameter",
        "AST#encapsed_string",
        "AST#expression_case",
        "AST#module",
        "AST#init_declarator",
        "AST#synchronized_statement",
        "AST#set_comprehension",
        "AST#static_initializer",
        "AST#lambda_expression",
        "AST#enumerator",
        "AST#subscript",
        "AST#dotted_name",
        "AST#preproc_ifdef",
        "AST#return_statement",
        "AST#generator_expression",
        "AST#parenthesized_type",
        "AST#debugger_statement",
        "AST#function_declaration",
        "AST#pattern_list",
        "AST#element_value_pair",
        "AST#function_definition",
        "AST#ifndef",
        "AST#if",
        "AST#jsx_opening_element",
        "AST#parameters",
        "AST#static",
        "AST#true",
        "AST#if_statement",
        "AST#not_operator",
        "AST#object_creation_expression",
        "AST#number_literal",
        "AST#spread_parameter",
        "AST#fallthrough_statement",
        "AST#constructor_constraint",
        "AST#type_identifier",
        "AST#try",
        "AST#access_specifier",
        "AST#element",
        "AST#while_modifier",
        "AST#statement_block",
        "AST#setter",
        "AST#array_access",
        "AST#anonymous_method_expression",
        "AST#namespace_definition",
        "AST#elif_directive",
        "AST#interpolation_alignment_clause",
        "AST#concatenated_string",
        "AST#join_clause",
        "AST#break",
        "AST#lambda_parameters",
        "AST#preproc_string_literal",
        "AST#await",
        "AST#modifiers",
        "AST#with_statement",
        "AST#undef",
        "AST#region_directive",
        "AST#compound_statement",
        "AST#begin",
        "AST#as_expression",
        "AST#field_identifier",
        "AST#preproc_message",
        "AST#switch_expression",
        "AST#pair_pattern",
        "AST#ref_value_expression",
        "AST#list_splat_pattern",
        "AST#field_initializer_list",
        "AST#annotated_type",
        "AST#type_parameter_list",
        "AST#jsx_self_closing_element",
        "AST#type_assertion_expression",
        "AST#block",
        "AST#label_name",
        "AST#declare_statement",
        "AST#update_expression",
        "AST#where_clause",
        "AST#shorthand_property_identifier",
        "AST#require_expression",
        "AST#simple_parameter",
        "AST#class_heritage",
        "AST#var",
        "AST#throw",
        "AST#local_declaration_statement",
        "AST#from_clause",
        "AST#enum_constant",
        "AST#array_initializer",
        "AST#size_of_expression",
        "AST#ms_call_modifier",
        "AST#name",
        "AST#parameter_declaration",
        "AST#selector_expression",
        "AST#if_directive",
        "AST#method_spec",
        "AST#integer",
        "AST#attribute_target_specifier",
        "AST#declaration_list",
        "AST#preproc_call",
        "AST#field_definition",
        "AST#catch_clause",
        "AST#base_list",
        "AST#none",
        "AST#constant_pattern",
        "AST#recursive_pattern",
        "AST#jsx_closing_element",
        "AST#checked_expression",
        "AST#receive_statement",
        "AST#tuple_expression",
        "AST#except_clause",
        "AST#catch_filter_clause",
        "AST#string",
        "AST#text",
        "AST#type_bound",
        "AST#namespace_definition_name",
        "AST#array_declarator",
        "AST#until_modifier",
        "AST#text_interpolation",
        "AST#modifier",
        "AST#endif",
        "AST#source_file",
        "AST#exception_variable",
        "AST#block_parameter",
        "AST#binary",
        "AST#operator_name",
        "AST#while_statement",
        "AST#local_function_statement",
        "AST#struct_type",
        "AST#instance_variable",
        "AST#boolean",
        "AST#index_expression",
        "AST#package_identifier",
        "AST#rescue",
        "AST#fixed_statement",
        "AST#named_type",
        "AST#abstract_parenthesized_declarator",
        "AST#echo_statement",
        "AST#decimal_floating_point_literal",
        "AST#variadic_declarator",
        "AST#preproc_elif",
        "AST#tuple_type",
        "AST#keyed_element",
        "AST#do_statement",
        "AST#let_clause",
        "AST#jsx_fragment",
        "AST#catch_declaration",
        "AST#function_static_declaration",
        "AST#parameter_modifier",
        "AST#const_spec",
        "AST#catch_type",
        "AST#singleton_class",
        "AST#null",
        "AST#ms_pointer_modifier",
        "AST#float",
        "AST#variable_name",
        "AST#binary_integer_literal",
        "AST#hash_key_symbol",
        "AST#pattern",
        "AST#for_in_statement",
        "AST#array_type",
        "AST#enhanced_for_statement",
        "AST#class_variable",
        "AST#relative_scope",
        "AST#preproc_def",
        "AST#range",
        "AST#enum_specifier",
        "AST#storage_class_specifier",
        "AST#self",
        "AST#property_identifier",
        "AST#escape_sequence",
        "AST#switch_block",
        "AST#constructor_body",
        "AST#formal_parameters",
        "AST#variadic_type_parameter_declaration",
        "AST#element_reference",
        "AST#optional_type_parameter_declaration",
        "AST#redo",
        "AST#lambda_capture_specifier",
        "AST#switch_section",
        "AST#noexcept",
        "AST#ifdef",
        "AST#primitive_type",
        "AST#group_clause",
        "AST#variable_declaration",
        "AST#member_call_expression",
        "AST#switch",
        "AST#type_descriptor",
        "AST#scoped_identifier",
        "AST#is_expression",
        "AST#interpolated_string_expression",
        "AST#case_statement",
        "AST#nested_identifier",
        "AST#global_variable",
        "AST#name_colon",
        "AST#instanceof_expression",
        "AST#field_access",
        "AST#generator_function_declaration",
        "AST#declaration",
        "AST#property_pattern_clause",
        "AST#delete_method_clause",
        "AST#spread_element",
        "AST#pragma_directive",
        "AST#class_body",
        "AST#statement_identifier",
        "AST#heredoc_body",
        "AST#initializer_pair",
        "AST#program",
        "AST#postfix_unary_expression",
        "AST#optional_type",
        "AST#dictionary_splat",
        "AST#type_case",
        "AST#field_declaration_list",
        "AST#warning_directive",
        "AST#object",
        "AST#method_definition",
        "AST#type_declaration",
        "AST#await_expression",
        "AST#destructor_name",
        "AST#computed_property_name",
        "AST#is",
        "AST#predefined_type",
        "AST#method_parameters",
        "AST#labeled_statement",
        "AST#catch",
        "AST#receiver_parameter",
        "AST#initializer_expression",
        "AST#cast_type",
        "AST#hash_splat_parameter",
        "AST#list_literal",
        "AST#print_intrinsic",
        "AST#property_initializer",
        "AST#preproc_defined",
        "AST#anonymous_function_creation_expression",
        "AST#type_parameter_constraints_clause",
        "AST#keyword_argument",
        "AST#switch_statement",
        "AST#interpolation_format_clause",
        "AST#boolean_type",
        "AST#variadic_unpacking",
        "AST#function_call_expression",
        "AST#regex",
        "AST#template_instantiation",
        "AST#stack_alloc_array_creation_expression",
        "AST#element_access_expression",
        "AST#ternary_expression",
        "AST#lambda_default_capture",
        "AST#or",
        "AST#object_assignment_pattern",
        # python diff
        "AST#keyword_separator",
        "AST#as_pattern_target",
        "AST#format_specifier",
        "AST#not in",
        "AST#as_pattern",
        "AST#format_expression",
        "AST#type_alias_statement",
        "AST#is not",
        # php diff
        "AST#static_modifier",
        "AST#use_declaration",
        "AST#nowdoc_body",
        "AST#readonly_modifier",
        "AST#anonymous_function",
        "AST#nowdoc",
        "AST#error_suppression_expression",
        "AST#anonymous_class",
        "AST#relative_name",
        "AST#exit_statement",
        # go diff
        "AST#literal_element",
        "AST#statement_list",
        "AST#var_spec_list",
        "AST#type_instantiation_expression",
        "AST#method_elem",
        "AST#type_elem",
        # ruby diff
        "AST#body_statement",
        "AST#block_body",
        # javascript diff
        "AST#function_expression",
    ]

    def __init__(self, args: DictConfig):
        self.args = args

        self.has_initialized = False
        self.test_ckpt = "last"
        # Validation, default one set
        self.validation_outputs = [[]]
        self.validation_outputs_special_tokens = [[]]
        self.validation_beam_outputs = [[]]
        self.validation_gold_special_tokens = [[]]
        self.validation_gold = [[]]
        # Test
        self.test_outputs = []
        self.test_beam_outputs = []
        self.test_gold = []

        # Default value is set to bootstrapping, used for full trainings.
        self.task_stage: TaskStage = TaskStage.FINETUNING
        self.emb_source: EmbSource = EmbSource.SELF
        if "fusion" in self.args.trainer:
            self.emb_source = EmbSource(self.args.trainer.fusion.emb_source)

    def get_train_steps(self, epoch=None) -> int:
        # if not provided, get max_train_epoch
        epoch = epoch if epoch else self.args.trainer.epoch

        # Assumes dataloders exists when this called
        num_update_steps_per_epoch = int(
            len(self.dataloaders["train"]) / self.args.trainer.accumulate_grad_batches
        )

        return epoch * num_update_steps_per_epoch

    def configure_metrics(self):
        passks, passk = {}, None
        self.metrics = {}

        def add_metrics(passks, metrics, dataset_config, subset):
            for m in dataset_config.metric:
                if m.startswith("pass"):
                    # add k values
                    k = int(m.split("@")[1])

                    if subset in passks:
                        if k not in passks[subset]:
                            passks[subset].append(k)
                    else:
                        passks[subset] = [k]
                else:
                    # add other metrics to the metrics
                    if subset in metrics:
                        metrics[subset].append(m)
                    else:
                        metrics[subset] = [m]

        dataset_config = self.args.dataset
        passk_rev = None
        if "train" in dataset_config:  # Act according to train dataset_config
            dataset_config = dataset_config.train

            # There are only one configuration, fix it later
            add_metrics(passks, self.metrics, dataset_config, "val")
            if "val" in self.args.dataset:
                add_metrics(passks, self.metrics, self.args.dataset.val, "val")
            for test_dataset_config in self.args.dataset.test.values():
                add_metrics(passks, self.metrics, test_dataset_config, "test")
            if len(passks) != 0:
                # Enable pass@k evaluation
                import os

                os.environ["HF_ALLOW_CODE_EVAL"] = "1"

                # Add pass@k metric keyword
                for subset in passks:
                    if subset in self.metrics:
                        self.metrics[subset].append("pass@k")
                    else:
                        self.metrics[subset] = ["pass@k"]

                # set 'k' values and language
                # there is only one configuration right now, fix later
                passk = partial(
                    load("evaluate-metric/code_eval").compute,
                    # code_eval_metric = evaluate.load("code_eval"), this is from odex
                    # language=dataset_config.pl_language,
                    k=passks["test"],
                    num_workers=32,
                    timeout=3.0,
                )
                # function accepts test cases first and references/golds later
                # also returns "a dictionary with granular results of each unit test."
                # which is ignored. Gold (g) here is assertions(test cases).
                passk_rev = lambda p, g: passk(references=g, predictions=p)[0]
        else:
            add_metrics(passks, self.metrics, dataset_config, "val")
            add_metrics(passks, self.metrics, dataset_config, "test")

        # prepare codebleu function
        calc_codebleu_lang = partial(calc_codebleu, lang=dataset_config.pl_language)
        metric_dict = {
            subset: {
                m: {
                    "em": ExactMatch(),
                    "bleu": google_bleu,
                    "codebleu": calc_codebleu_lang,
                    "pass@k": passk_rev,
                }[m]
                for m in ms
            }
            for subset, ms in self.metrics.items()
        }

        self.evaluator = lambda preds, golds, prefix: {
            f"{prefix}_{m}": f(preds, golds) for m, f in metric_dict[prefix].items()
        }

    def configure_special_tokens(self):
        generation_config = GenerationConfig.from_model_config(self.model.config) if hasattr(self.model, "config") else GenerationConfig()
        model_name = self.args.model.name.lower()

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id

        if model_name == "bart":
            generation_config.bos_token_id = self.tokenizer.bos_token_id
            self.model.config.bos_token_id = self.tokenizer.bos_token_id

            generation_config.decoder_start_token_id = self.tokenizer.bos_token_id
            self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id

            # generation_config.forced_bos_token_id = None
            # generation_config.forced_eos_token_id = None
        elif model_name == "plbart":
            # ['__javascript__', '__go__', '__php__', '__ruby__', '__c__', '__cpp__', '__csharp__']
            if self.args.dataset.pl_language not in ['python', 'java']:
                raise NotImplementedError(f"Language {self.args.dataset.pl_language} is not added to the PLBART model yet!")
            if self.args.dataset.pl_feed == 'input':
                input_lang = f"__{self.args.dataset.pl_language}__"
                output_lang = '__en_XX__'
            elif self.args.dataset.pl_feed == 'output':
                input_lang = '__en_XX__'
                output_lang = f"__{self.args.dataset.pl_language}__"
            else:
                ## inout option
                raise NotImplementedError("inout option for pl_language is not implemented")

            generation_config.bos_token_id = self.tokenizer.lang_code_to_id[input_lang]
            self.model.config.bos_token_id = self.tokenizer.lang_code_to_id[input_lang]
            # self.tokenizer.bos_token = input_lang # dont add to the input size for now?

            generation_config.decoder_start_token_id = self.tokenizer.lang_code_to_id[output_lang]
            self.model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[output_lang]
        elif model_name == "codet5":
            # 0: <pad> 1: <s>
            generation_config.bos_token_id = 1
            self.model.config.bos_token_id = 1
        elif model_name == "t5":
            # T5 does not use started tokens
            generation_config.bos_token_id = None
        elif (
            "bert" in model_name.lower()
            or model_name.lower() == "unixcoder"
        ):
            is_bert = "bert" == model_name.lower()
            if is_bert:
                self.model.config.eos_token_id = self.tokenizer.sep_token_id
                generation_config.eos_token_id = self.tokenizer.sep_token_id
                self.tokenizer.add_special_tokens({"bos_token": "[CLS]"})

                self.model.config.bos_token_id = self.tokenizer.bos_token_id
                generation_config.bos_token_id = self.tokenizer.bos_token_id
            # [SEP] is broken with BERT, we use [CLS] as start token
            dstidx = (
                # self.tokenizer.sep_token_id if is_bert else
                generation_config.bos_token_id
            )
            self.model.config.decoder_start_token_id = dstidx
            generation_config.decoder_start_token_id = dstidx

        self.model.generation_config = generation_config
        self.dataset.generation_config = generation_config

    def on_embeddings_called(self, input_ids):
        # Add debugging info if needed
        pass

    def standardize_embeddings(self, new_embeddings):
        # Update model architecture so all models follow the same structure
        # i.e. Generating ED from E models had different structures
        # Setup observer for embeddings calls
        def wrap_embedding_forward(original_forward):
            def wrapped_forward(input_ids):
                if hasattr(self, 'on_embeddings_called'):
                    self.on_embeddings_called(input_ids)
                result = original_forward(input_ids)
                return result
            return wrapped_forward

        # Wrap the new_embeddings forward method
        if hasattr(new_embeddings, 'forward'):
            new_embeddings.forward = wrap_embedding_forward(
                new_embeddings.forward
            )

        # Set embeddings
        # Handle different model architectures
        if hasattr(self.model, 'shared'):
            # T5/CodeT5 AND modified ED models: Update shared embeddings
            self.model.shared = new_embeddings
            self.model.encoder.embed_tokens = self.model.shared  # For T5-style
            self.model.decoder.embed_tokens = self.model.shared  # For T5-style
            # For ED models, also update the embeddings directly
            self.model.encoder.set_input_embeddings(self.model.shared)
            self.model.decoder.set_input_embeddings(self.model.shared)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'shared'):
            # PLBART-style: model.model.shared
            self.model.model.shared = new_embeddings
            self.model.model.encoder.embed_tokens = self.model.model.shared
            self.model.model.decoder.embed_tokens = self.model.model.shared
        elif hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
            # Legacy ED models: Update both separately
            self.model.encoder.set_input_embeddings(new_embeddings)
            self.model.decoder.set_input_embeddings(new_embeddings)
        else:
            # Single-stack models
            self.model.set_input_embeddings(new_embeddings)

    def generate_text_report(self, preds, golds, metadata, report_artifact):
        num_examples = len(golds) if report_artifact == "all" else report_artifact

        report = "".join([
            f"{golds[j]}\n{preds[j][0] if isinstance(preds[j], list) else preds[j]}\n\n"
            for j in range(num_examples)
        ])

        return report + metadata

    def gen_txt(self, pred_group, gold_group, report_artifact, generate_html=False):
        for i, (preds, golds) in enumerate(zip(pred_group, gold_group)):
            metadata = (
                self.test_dataloader_names[i] + "\n"
                if self.current_epoch == 0
                else ""
            )

            yield i, self.generate_text_report(preds, golds, metadata, report_artifact)

    def evaluate(self, pred_group, gold_group, report_artifact=5, is_beam=False):
        prefix = self.trainer.state.stage.dataloader_prefix
        context = self.trainer.state.stage.lower()

        if "pass@k" in self.metrics[prefix]:
            pred_group = [[[v.strip() for v in a] for a in p] for p in pred_group]
        else:
            pred_group = [[v.strip() for v in p] for p in pred_group]
        gold_group = [[v.strip() for v in g] for g in gold_group]

        results = {}
        for i, (preds, golds) in enumerate(zip(pred_group, gold_group)):
            for name, res in self.evaluator(preds, golds, prefix).items():
                # pass@k returns as a dict but our key is pass@k
                if isinstance(res, dict):
                    results.update(
                        {sub_name: sub_res for sub_name, sub_res in res.items()}
                    )
                else:
                    results[name] = res

        # Report metrics
        if context == "test":
            results = {k.split("_")[1]: v for k, v in results.items()}

            # Either report only as test when test set is from same dataset
            # or add test dataset name to be able to differentiate
            ctx = (
                context
                if self.test_dataloader_names[i] == self.train_dataloader_name
                else self.test_dataloader_names[i]
            )
            # If tested with last state (or any other set by __main__),
            # add that to the naming...
            # If tested with best checkpoint, then keep it as is
            if self.test_ckpt != "best":
                ctx = f"{ctx}_{self.test_ckpt}"
            if is_beam:
                ctx = f"{ctx}_beam"

            # Track results with logger - works for both Aim and MLflow
            if hasattr(self.logger.experiment, "track"):
                # Aim logger
                self.logger.experiment.track(
                    results,
                    context={"subset": ctx},
                    epoch=self.trainer.current_epoch,
                )
            elif hasattr(self.logger.experiment, "log_metrics"):
                # MLflow logger - log metrics with prefix
                prefixed_results = {f"{ctx}_{k}": v for k, v in results.items()}
                self.logger.experiment.log_metrics(
                    prefixed_results, step=self.trainer.current_epoch
                )
        else:
            self.log_dict(
                results,
                on_step=False,
                logger=True,
            )

        # Check if HTML generation is enabled (from config)
        generate_html = getattr(self.args, 'generate_html_reports', False)

        for i, txt in self.gen_txt(pred_group, gold_group, report_artifact, generate_html):
            # Determine file extension based on format
            file_ext = "html" if generate_html else "txt"
            filename = f"generation_{context}_{i}.{file_ext}"

            # Track generation artifacts if logger supports it
            if hasattr(self.logger.experiment, "log_text"):
                # MLflow logger - log as text/HTML artifacts
                self.logger.experiment.log_text(txt, filename)
            elif hasattr(self.logger.experiment, "track"):
                # Aim logger
                if generate_html:
                    # For HTML, track as text with HTML context
                    from aim import Text
                    self.logger.experiment.track(
                        Text(txt),
                        name="generation_html",
                        context={"subset": context, "format": "html"},
                        epoch=self.current_epoch,
                    )
                else:
                    # Original Aim Text wrapper for plain text
                    from aim import Text
                    self.logger.experiment.track(
                        Text(txt),
                        name="generation",
                        context={"subset": context},
                        epoch=self.current_epoch,
                    )
                    if context != "test":
                        try:
                            golds = self.validation_gold_special_tokens[i]
                            preds = self.validation_outputs_special_tokens[i]
                            # Also log with specialTokens context for validation
                            report = "".join([
                                f"{golds[j]}\n{preds[j][0] if isinstance(preds[j], list) else preds[j]}\n\n"
                                for j in range(5)
                            ])
                            self.logger.experiment.track(
                                Text(report),
                                name="generation",
                                context={"subset": context + "_specialTokens"},
                                epoch=self.current_epoch,
                            )
                        except Exception as e:
                            pass
            elif hasattr(self.logger.experiment, "log_artifact"):
                # MLflow logger - alternative method for artifacts
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{file_ext}', delete=False) as f:
                    f.write(txt)
                    temp_path = f.name

                try:
                    self.logger.experiment.log_artifact(temp_path, filename)
                finally:
                    os.unlink(temp_path)

        return results
