"""
Questionnaire Data Transformer
Converts UI questionnaire data to RAG-friendly format using YAML labels for context
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class QuestionMapping:
    """Represents a question mapping from YAML configuration."""
    key: str
    label: str
    field_type: str
    options: Optional[List[str]] = None
    section_title: Optional[str] = None
    group_title: Optional[str] = None

class QuestionnaireTransformer:
    """Transforms UI questionnaire data using YAML configuration for better context."""
    
    def __init__(self, yaml_config_path: str = "../UI-Experimentation5/public/config/onboarding.yaml"):
        """Initialize with path to YAML configuration file."""
        self.yaml_path = Path(yaml_config_path)
        self.question_mappings: Dict[str, QuestionMapping] = {}
        self._load_yaml_config()
    
    def _load_yaml_config(self):
        """Load and parse YAML configuration to build question mappings."""
        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Parse all tabs, sections, groups, and fields
            for tab in config.get('tabs', []):
                for section in tab.get('sections', []):
                    section_title = section.get('title', '')
                    
                    for group in section.get('groups', []):
                        group_title = group.get('title', '')
                        
                        for field in group.get('fields', []):
                            key = field.get('key')
                            if key:
                                # Extract options if they exist
                                options = field.get('options', [])
                                if options and isinstance(options[0], dict):
                                    # Handle options with value/label structure
                                    options = [opt.get('label', opt.get('value', str(opt))) for opt in options]
                                
                                self.question_mappings[key] = QuestionMapping(
                                    key=key,
                                    label=field.get('label', key),
                                    field_type=field.get('type', 'text'),
                                    options=options if options else None,
                                    section_title=section_title,
                                    group_title=group_title
                                )
                                
            print(f"✅ Loaded {len(self.question_mappings)} question mappings from YAML")
            
        except FileNotFoundError:
            print(f"⚠️  YAML config file not found: {self.yaml_path}")
            print("   Using fallback mode - keys will be used as labels")
        except Exception as e:
            print(f"⚠️  Error loading YAML config: {e}")
            print("   Using fallback mode - keys will be used as labels")
    
    def transform_questionnaire_data(self, ui_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform UI questionnaire data into RAG-friendly format with full context."""
        
        if not ui_data or 'data' not in ui_data:
            return {}
        
        values = ui_data['data'].get('values', {})
        
        # Create comprehensive profile with full question context
        transformed_profile = {
            # Meta information
            "_meta": {
                "transformation_source": "UI questionnaire",
                "total_fields_processed": len(values),
                "yaml_mappings_available": len(self.question_mappings) > 0
            }
        }
        
        # Transform each field with full context
        organized_data = self._organize_data_by_sections(values)
        
        # Add organized sections
        transformed_profile.update(organized_data)
        
        # Create summary for quick reference
        transformed_profile["_summary"] = self._create_summary(values)
        
        return transformed_profile
    
    def _organize_data_by_sections(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Organize data by sections with full question context."""
        
        sections = {}
        
        for key, value in values.items():
            if not value:  # Skip empty values
                continue
                
            mapping = self.question_mappings.get(key)
            
            if mapping:
                # Use section title or create default section
                section_key = self._normalize_section_name(mapping.section_title or "General")
                
                if section_key not in sections:
                    sections[section_key] = {}
                
                # Create human-readable field entry
                field_entry = self._format_field_value(mapping, value)
                sections[section_key][key] = field_entry
            else:
                # Fallback for unmapped keys
                if "unmapped_fields" not in sections:
                    sections["unmapped_fields"] = {}
                sections["unmapped_fields"][key] = {
                    "question": key.replace('_', ' ').replace('-', ' ').title(),
                    "answer": value,
                    "raw_key": key
                }
        
        return sections
    
    def _format_field_value(self, mapping: QuestionMapping, value: Any) -> Dict[str, Any]:
        """Format a field value with full context."""
        
        formatted_value = {
            "question": mapping.label,
            "answer": value,
            "field_type": mapping.field_type,
            "raw_key": mapping.key
        }
        
        # Add section context if available
        if mapping.section_title:
            formatted_value["section"] = mapping.section_title
        if mapping.group_title:
            formatted_value["group"] = mapping.group_title
        
        # Handle different field types
        if mapping.field_type == "checkboxes" and isinstance(value, list):
            formatted_value["selected_options"] = value
            formatted_value["available_options"] = mapping.options
            formatted_value["answer_summary"] = f"Selected {len(value)} options: {', '.join(value)}"
        
        elif mapping.field_type == "radio" and mapping.options:
            formatted_value["selected_option"] = value
            formatted_value["available_options"] = mapping.options
        
        return formatted_value
    
    def _create_summary(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Create a quick summary of key information."""
        
        summary = {}
        
        # Key organization info
        org_fields = ['legalOrgName', 'industry', 'orgSize', 'primaryCustomers', 'operatingRegions']
        org_info = {}
        for field in org_fields:
            if field in values and values[field]:
                mapping = self.question_mappings.get(field)
                label = mapping.label if mapping else field.replace('_', ' ').title()
                org_info[label] = values[field]
        
        if org_info:
            summary["Organization Profile"] = org_info
        
        # Key product info  
        product_fields = ['productName', 'elevatorPitch', 'useCases', 'paradigm', 'primaryUsers']
        product_info = {}
        for field in product_fields:
            if field in values and values[field]:
                mapping = self.question_mappings.get(field)
                label = mapping.label if mapping else field.replace('_', ' ').title()
                product_info[label] = values[field]
        
        if product_info:
            summary["Product Information"] = product_info
        
        # Key values and compliance
        values_fields = ['orgValues', 'complianceDrivers', 'riskAppetite']
        values_info = {}
        for field in values_fields:
            if field in values and values[field]:
                mapping = self.question_mappings.get(field)
                label = mapping.label if mapping else field.replace('_', ' ').title()
                values_info[label] = values[field]
        
        if values_info:
            summary["Values & Compliance"] = values_info
        
        return summary
    
    def _normalize_section_name(self, section_title: str) -> str:
        """Normalize section names for consistent organization."""
        return section_title.lower().replace(' ', '_').replace('&', 'and').replace('/', '_')
    
    def get_formatted_context_string(self, transformed_data: Dict[str, Any]) -> str:
        """Create a formatted string representation for RAG context."""
        
        if not transformed_data:
            return "No questionnaire data available."
        
        context_parts = []
        
        # Add summary first
        if "_summary" in transformed_data:
            context_parts.append("=== QUESTIONNAIRE SUMMARY ===")
            summary = transformed_data["_summary"]
            
            for section_name, section_data in summary.items():
                context_parts.append(f"\n{section_name}:")
                if isinstance(section_data, dict):
                    for question, answer in section_data.items():
                        context_parts.append(f"  • {question}: {self._format_answer_for_context(answer)}")
                else:
                    context_parts.append(f"  {section_data}")
        
        # Add detailed sections
        context_parts.append("\n\n=== DETAILED RESPONSES ===")
        
        for section_key, section_data in transformed_data.items():
            if section_key.startswith('_'):  # Skip meta fields
                continue
                
            section_title = section_key.replace('_', ' ').title()
            context_parts.append(f"\n{section_title}:")
            
            if isinstance(section_data, dict):
                for field_key, field_data in section_data.items():
                    if isinstance(field_data, dict) and 'question' in field_data:
                        question = field_data['question']
                        answer = field_data.get('answer', 'No answer provided')
                        context_parts.append(f"  • {question}: {self._format_answer_for_context(answer)}")
                    else:
                        context_parts.append(f"  • {field_key}: {field_data}")
        
        return "\n".join(context_parts)
    
    def _format_answer_for_context(self, answer: Any) -> str:
        """Format an answer for inclusion in context string."""
        if isinstance(answer, list):
            if len(answer) == 0:
                return "None selected"
            return f"{', '.join(map(str, answer))} ({len(answer)} selected)"
        elif isinstance(answer, str):
            return answer if answer.strip() else "Not specified"
        elif answer is None:
            return "Not specified"
        else:
            return str(answer)

# Global transformer instance
questionnaire_transformer = QuestionnaireTransformer()

def transform_ui_data_to_rag_profile(ui_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to transform UI data for RAG system."""
    return questionnaire_transformer.transform_questionnaire_data(ui_data)

def get_questionnaire_context_string(ui_data: Dict[str, Any]) -> str:
    """Get formatted context string for RAG system."""
    transformed = questionnaire_transformer.transform_questionnaire_data(ui_data)
    return questionnaire_transformer.get_formatted_context_string(transformed)


if __name__ == "__main__":
    # Test the transformer
    sample_data = {
        "data": {
            "values": {
                "legalOrgName": "Test Company",
                "industry": "Finance",
                "orgValues": ["Fairness/Non-discrimination", "Privacy/Security"],
                "useCases": "Building AI-powered fraud detection"
            }
        }
    }
    
    transformed = transform_ui_data_to_rag_profile(sample_data)
    context_string = get_questionnaire_context_string(sample_data)
    
    print("=== TRANSFORMED DATA ===")
    print(json.dumps(transformed, indent=2))
    
    print("\n=== CONTEXT STRING ===")
    print(context_string)
