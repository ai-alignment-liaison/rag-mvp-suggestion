"""
Quick test of the questionnaire transformer with real data
"""

import json
from questionnaire_transformer import transform_ui_data_to_rag_profile, get_questionnaire_context_string

# Load real questionnaire data
def test_transformer():
    # Sample data based on the JSON structure we saw
    sample_data = {
        "version": 1,
        "data": {
            "values": {
                "legalOrgName": "Regenerative Serenity Bank",
                "publicWebsiteUrl": "https://www.levelingupwithxai.com/",
                "operatingRegions": ["United States", "EU/EEA", "UK"],
                "industry": "Finance",
                "orgSize": "1k+",
                "primaryCustomers": "SMBs",
                "riskAppetite": "Balanced",
                "redTeamReview": "Yes",
                "orgValues": [
                    "Fairness/Non-discrimination",
                    "Privacy/Security",
                    "Transparency/Auditability"
                ],
                "complianceDrivers": [
                    "EU AI Act",
                    "ECOA/Reg B"
                ],
                "productName": "AI Fraud Detection System",
                "elevatorPitch": "AI-powered fraud detection for banking",
                "useCases": "Real-time fraud detection and risk assessment",
                "paradigm": "RAG over proprietary data"
            },
            "step": 0,
            "uploadedFiles": [],
            "ts": 1757396639569
        }
    }
    
    print("🧪 Testing Questionnaire Transformer")
    print("=" * 50)
    
    # Transform the data
    transformed = transform_ui_data_to_rag_profile(sample_data)
    
    print("✅ Transformation successful!")
    print(f"Sections created: {len([k for k in transformed.keys() if not k.startswith('_')])}")
    
    if "_summary" in transformed:
        print("\n📋 Summary sections:")
        for section_name in transformed["_summary"].keys():
            print(f"  • {section_name}")
    
    # Get context string
    context_string = get_questionnaire_context_string(sample_data)
    
    print(f"\n📝 Context string length: {len(context_string)} characters")
    print("\n🎯 Sample context preview:")
    print("-" * 30)
    print(context_string[:500] + "..." if len(context_string) > 500 else context_string)
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_transformer()
