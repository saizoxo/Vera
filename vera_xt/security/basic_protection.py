#/storage/emulated/0/Vxt/Vxt/vera_xt/security/basic_protection.py
#!/usr/bin/env python3
"""
Basic Protection System - Intruder detection and identity verification
Learns user patterns to detect unauthorized access
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class BasicProtection:
    def __init__(self, security_dir: str = "Memory_Data"):
        self.security_dir = Path(security_dir)
        self.security_dir.mkdir(exist_ok=True)
        
        # User identity patterns (learned, not hardcoded)
        self.user_patterns = {
            "communication_style": {},
            "common_topics": set(),
            "time_patterns": {},
            "response_patterns": set(),
            "personal_references": set()
        }
        
        # Security tracking
        self.access_logs = []
        self.suspicious_activities = []
        self.identity_confirmed = False
        self.session_trust_level = 0.0  # 0.0 to 1.0
        
        # Personal verification data
        self.personal_verification = {
            "known_projects": set(),
            "known_files": set(),
            "known_activities": set(),
            "behavioral_patterns": {}
        }
        
        print("🛡️  Basic Protection System initialized")
        print("💡 Learns user patterns to detect intruders")
    
    def update_user_patterns(self, user_input: str, context: Dict[str, Any] = None):
        """Update learned user patterns from interaction"""
        # Update communication style
        word_count = len(user_input.split())
        if word_count < 10:
            self.user_patterns["communication_style"]["direct"] = \
                self.user_patterns["communication_style"].get("direct", 0) + 1
        elif word_count > 50:
            self.user_patterns["communication_style"]["detailed"] = \
                self.user_patterns["communication_style"].get("detailed", 0) + 1
        
        # Update common topics
        if context:
            input_type = context.get("input_type", "general")
            self.user_patterns["common_topics"].add(input_type)
            
            # Update time patterns
            hour = datetime.now().hour
            self.user_patterns["time_patterns"][hour] = \
                self.user_patterns["time_patterns"].get(hour, 0) + 1
        
        # Update response patterns (what user typically asks about)
        if any(word in user_input.lower() for word in ["code", "python", "programming"]):
            self.user_patterns["response_patterns"].add("technical_help")
        elif any(word in user_input.lower() for word in ["plan", "organize", "schedule"]):
            self.user_patterns["response_patterns"].add("planning_help")
        elif any(word in user_input.lower() for word in ["learn", "study", "understand"]):
            self.user_patterns["response_patterns"].add("learning_help")
    
    def verify_identity(self, query: str) -> Dict[str, Any]:
        """Verify if the user is legitimate based on learned patterns"""
        verification_result = {
            "is_likely_user": True,
            "confidence_level": 0.8,  # Default high, adjusted based on verification
            "suspicious_indicators": [],
            "verification_details": {}
        }
        
        # Check for personal references that don't match learned patterns
        personal_indicators = [
            "my project", "our plan", "i started", "we did", "my file", 
            "my work", "our code", "my assignment", "our task"
        ]
        
        query_lower = query.lower()
        
        # Check if asking about personal things user never mentioned
        for indicator in personal_indicators:
            if indicator in query_lower:
                # Check if this specific personal reference matches learned patterns
                words = query_lower.split()
                for word in words:
                    if word in ["project", "file", "code", "work", "assignment", "task"]:
                        # This could be suspicious if not in learned personal references
                        if word not in self.personal_verification["known_activities"]:
                            verification_result["suspicious_indicators"].append(
                                f"Unknown personal reference: {word}"
                            )
                            verification_result["confidence_level"] *= 0.7  # Reduce confidence
        
        # Check communication style consistency
        if len(query.split()) < 5:
            # User typically uses longer messages
            if self.user_patterns["communication_style"].get("detailed", 0) > 5:
                verification_result["suspicious_indicators"].append("Communication style mismatch")
                verification_result["confidence_level"] *= 0.8
        
        # Check for unusual timing (if user typically active at certain hours)
        current_hour = datetime.now().hour
        if current_hour not in self.user_patterns["time_patterns"]:
            # User rarely active at this time
            if len(self.user_patterns["time_patterns"]) > 2:  # If we have enough data
                verification_result["suspicious_indicators"].append("Unusual activity time")
                verification_result["confidence_level"] *= 0.9
        
        # Final confidence adjustment
        verification_result["is_likely_user"] = verification_result["confidence_level"] > 0.5
        verification_result["trust_level"] = verification_result["confidence_level"]
        
        return verification_result
    
    def handle_suspicious_query(self, query: str, verification_result: Dict[str, Any]) -> str:
        """Handle queries that seem suspicious"""
        if verification_result["is_likely_user"]:
            return "normal_response"  # Proceed normally
        
        # Log suspicious activity
        self.suspicious_activities.append({
            "timestamp": time.time(),
            "query": query,
            "confidence": verification_result["confidence_level"],
            "indicators": verification_result["suspicious_indicators"]
        })
        
        # Decide response based on trust level
        if verification_result["confidence_level"] < 0.3:
            return "security_challenge"  # Ask verification question
        elif verification_result["confidence_level"] < 0.6:
            return "limited_response"    # Provide generic help
        else:
            return "normal_response"     # Mostly normal with caution
    
    def security_challenge(self, query: str) -> str:
        """Provide a security challenge to verify identity"""
        # Generate a challenge based on learned personal patterns
        if self.personal_verification["known_projects"]:
            project = list(self.personal_verification["known_projects"])[0]
            return f"I remember we worked on {project}. Can you tell me what the main goal was?"
        elif self.user_patterns["common_topics"]:
            topic = list(self.user_patterns["common_topics"])[0]
            return f"We often discuss {topic}. What was the last {topic} project we worked on?"
        else:
            return "I'd like to verify your identity. Can you tell me about something we've worked on together before?"
    
    def provide_limited_response(self, query: str) -> str:
        """Provide a limited, safe response for low-trust situations"""
        return "I can help with general questions and learning, but I'll need to verify your identity for personal information. What general topic would you like assistance with?"
    
    def process_query_with_security(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process query with security verification"""
        # Verify identity
        verification = self.verify_identity(query)
        
        # Handle based on verification result
        response_type = self.handle_suspicious_query(query, verification)
        
        result = {
            "query": query,
            "verification": verification,
            "response_type": response_type,
            "security_level": "normal" if verification["is_likely_user"] else "protected"
        }
        
        # Log the access
        self.access_logs.append({
            "timestamp": time.time(),
            "query": query,
            "verification_result": verification["is_likely_user"],
            "response_type": response_type
        })
        
        # Update trust level
        self.session_trust_level = verification["confidence_level"]
        
        return result
    
    def add_personal_verification_data(self, category: str, item: str):
        """Add verified personal data for future reference"""
        if category == "project":
            self.personal_verification["known_projects"].add(item)
        elif category == "file":
            self.personal_verification["known_files"].add(item)
        elif category == "activity":
            self.personal_verification["known_activities"].add(item)
        
        # Update behavioral patterns
        if category not in self.personal_verification["behavioral_patterns"]:
            self.personal_verification["behavioral_patterns"][category] = []
        self.personal_verification["behavioral_patterns"][category].append(item)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        return {
            "trust_level": self.session_trust_level,
            "total_accesses": len(self.access_logs),
            "suspicious_activities_count": len(self.suspicious_activities),
            "known_user_patterns": {
                "communication_styles": dict(self.user_patterns["communication_style"]),
                "common_topics": list(self.user_patterns["common_topics"]),
                "time_patterns": dict(self.user_patterns["time_patterns"]),
                "response_patterns": list(self.user_patterns["response_patterns"])
            },
            "personal_verification_data": {
                "known_projects": list(self.personal_verification["known_projects"]),
                "known_files": list(self.personal_verification["known_files"]),
                "known_activities": list(self.personal_verification["known_activities"]),
                "behavioral_patterns": dict(self.personal_verification["behavioral_patterns"])
            },
            "recent_suspicious_activities": self.suspicious_activities[-5:]  # Last 5
        }
    
    def reset_session_trust(self):
        """Reset session trust level (for new sessions)"""
        self.session_trust_level = 0.0
        self.identity_confirmed = False
    
    def confirm_identity(self):
        """Manually confirm identity (when user proves who they are)"""
        self.identity_confirmed = True
        self.session_trust_level = 1.0
