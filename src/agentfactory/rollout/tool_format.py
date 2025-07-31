"""Tool format abstraction for different LLM tool calling protocols.

This module provides a unified system for:
- System message generation (tool schemas -> prompt)
- Tool call parsing (LLM response -> structured calls)  
- Tool response formatting (tool output -> LLM input)
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import re
import textwrap

# ============================================================================
# Base Tool Format Interface
# ============================================================================

class ToolFormat(ABC):
    """Abstract base class for tool calling formats"""
    
    @abstractmethod
    def build_system_message(
        self,
        tool_definitions: List[Dict[str, Any]],
        max_calls_per_round: int = 1,
        include_image_instructions: bool = True,
    ) -> str:
        """Generate system message content for tool descriptions"""
        pass
    
    @abstractmethod
    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response text
        
        Returns:
            List of tool calls in format [{"tool": "tool_name", "arguments": {...}}, ...]
        """
        pass
    
    def format_tool_response(
        self, 
        tool_name: str,
        tool_output: List[Dict[str, Any]],
        structured_output: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Format tool response for LLM consumption
        
        Default implementation returns output as-is.
        Override for custom formatting.
        """
        return tool_output
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Format identifier"""
        pass

    @property
    @abstractmethod
    def stop_sequences(self) -> List[str]:
        """Stop sequences for the tool format"""
        pass

# ============================================================================
# XML-based Tool Format (Default Implementation)
# ============================================================================

class AgentFactoryToolFormat(ToolFormat):
    """AgentFactory tool format using AFML-style syntax"""
    
    @property
    def name(self) -> str:
        return "agentfactory"
    
    @property
    def stop_sequences(self) -> List[str]:
        return ["</afml:function_calls>"]
    
    def build_system_message(
        self,
        tool_definitions: List[Dict[str, Any]],
        max_calls_per_round: int = 1,
        include_image_instructions: bool = True,
    ) -> str:
        """Build AgentFactory-style system message"""
        
        def format_tool_parameters(input_schema: Dict[str, Any]) -> Dict[str, Any]:
            """Extract and format tool parameters from tool schema"""
            parameter_schema = {}
            required_fields = input_schema.get('required', [])
            
            for param_name, param_spec in input_schema['properties'].items():
                param_info = {}
                if 'description' in param_spec:
                    param_info['description'] = param_spec['description']
                if 'type' in param_spec:
                    param_info['type'] = param_spec['type']
                if 'default' in param_spec:
                    param_info['default'] = param_spec['default']
                param_info['required'] = param_name in required_fields
                parameter_schema[param_name] = param_info
            
            return parameter_schema

        # Build tool descriptions
        tool_descriptions = []
        for tool in tool_definitions:
            tool_desc = textwrap.dedent(f"""
                <function>
                <name>{tool['name']}</name>
                <description>{tool['description']}</description>
                <parameters>{format_tool_parameters(tool['inputSchema'])}</parameters>
                <max_calls>{tool['usage']['max_calls']}</max_calls>
                </function>
            """).strip()
            tool_descriptions.append(tool_desc)
        
        tools_section = '\n\n'.join(tool_descriptions)
        
        # Build calling instructions based on max calls
        if max_calls_per_round > 1:
            calling_instructions = textwrap.dedent("""
                In this environment you have access to a set of tools you can use to answer the user's question.
                You can invoke functions by writing a "<afml:function_calls>" block like the following as part of your reply to the user:
                <afml:function_calls>
                <invoke name="$FUNCTION_NAME">
                <parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</parameter>
                ...
                </invoke>
                <invoke name="$FUNCTION_NAME2">
                ...
                </invoke>
                </afml:function_calls>
                String and scalar parameters should be specified as is, while lists and objects should use JSON format.
                If you intend to call multiple tools and there are no dependencies between the calls, make all of the independent calls in the same <afml:function_calls></afml:function_calls> block.
            """).strip()
        else:
            calling_instructions = textwrap.dedent("""
                In this environment you have access to a set of tools you can use to answer the user's question.
                You can invoke functions by writing a "<afml:function_calls>" block like the following as part of your reply to the user:
                <afml:function_calls>
                <invoke name="$FUNCTION_NAME">
                <parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</parameter>
                ...
                </invoke>
                </afml:function_calls>
                String and scalar parameters should be specified as is, while lists and objects should use JSON format.
                You can only invoke one tool at a time in each function call block.
            """).strip()

        # Build complete message parts
        message_parts = [
            "\n",
            calling_instructions,
            "",
            "Here are the functions available:",
            "<functions>",
            tools_section,
            "</functions>"
        ]
        
        # Add image processing instructions if needed
        if include_image_instructions:
            image_instructions = textwrap.dedent("""
                
                Regarding image processing:
                Images submitted by users or generated by tools will be automatically resized and presented in the following format:
                <image path=... display_size=WxH>
                While display images are downsampled to optimize computational resources, you will maintain access to the original full-resolution images when operating within the tool environment. This ensures efficient display while preserving complete image data for any analytical or processing tasks you perform.
            """).rstrip()
            message_parts.append(image_instructions)
        
        return "\n".join(message_parts)
    
    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse XML-style tool calls from LLM response"""
        pattern = r'\<afml:function_calls\>(.*?)\</afml:function_calls\>'
        function_blocks = re.findall(pattern, text, re.DOTALL)
        
        results = []
        for block in function_blocks:
            invoke_pattern = r'\<invoke name="([^"]+)"\>(.*?)\</invoke\>'
            invocations = re.findall(invoke_pattern, block, re.DOTALL)
            
            for function_name, params_text in invocations:
                param_pattern = r'\<parameter name="([^"]+)"\>(.*?)\</parameter\>'
                params = re.findall(param_pattern, params_text, re.DOTALL)
                
                function_call = {
                    "tool": function_name,
                    "arguments": {name: value.strip() for name, value in params}
                }
                results.append(function_call)
        
        return results

# ============================================================================
# Format Registry and Factory
# ============================================================================

class ToolFormatRegistry:
    """Registry for tool formats"""
    
    _formats: Dict[str, ToolFormat] = {}
    
    @classmethod
    def register(cls, format_instance: ToolFormat):
        """Register a tool format"""
        cls._formats[format_instance.name] = format_instance
    
    @classmethod
    def get(cls, name: str) -> Optional[ToolFormat]:
        """Get a tool format by name"""
        return cls._formats.get(name)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available format names"""
        return list(cls._formats.keys())

# Register built-in formats
ToolFormatRegistry.register(AgentFactoryToolFormat())

def create_tool_format(name: str) -> ToolFormat:
    """Factory function to create tool format instance"""
    tool_format = ToolFormatRegistry.get(name)
    if tool_format is None:
        available = ToolFormatRegistry.list_available()
        raise ValueError(f"Unknown tool format: {name}. Available: {available}")
    return tool_format