"""
Aircraft Safety Analysis System - FIXED VERSION
Analyzes landing, taxi, and takeoff videos for safety issues and damage
Generates comprehensive PDF reports with safety grading
"""

import os
from dotenv import load_dotenv
load_dotenv()

import base64
from typing import List, Dict, Optional, Tuple
import cv2
from datetime import timedelta, datetime
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas


class AircraftSafetyAnalyzer:
    """Analyze aircraft videos for safety issues and generate reports."""
    
    # Safety grading system
    SAFETY_GRADES = {
        "OK": {
            "color": colors.green,
            "description": "No issues detected - aircraft safe for operation"
        },
        "CAUTION": {
            "color": colors.yellow,
            "description": "Minor issues requiring inspection before next flight"
        },
        "REPAIR_NEEDED": {
            "color": colors.orange,
            "description": "Damage or issues requiring repair before next operation"
        },
        "IMMEDIATE_GROUNDING": {
            "color": colors.red,
            "description": "Critical safety issue - aircraft must be grounded immediately"
        }
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """Initialize the aircraft safety analyzer."""
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            max_tokens=4096,
            temperature=0.3  # Lower temperature for more consistent technical analysis
        )
        self.output_parser = StrOutputParser()
    
    def extract_frames(
        self,
        video_path: str,
        num_frames: int = 20
    ) -> List[Dict]:
        """Extract frames from video with high detail for damage detection."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        frames_data = []
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Use high quality JPEG encoding for better damage detection
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_bytes = buffer.tobytes()
                base64_image = base64.b64encode(frame_bytes).decode('utf-8')
                
                timestamp = idx / fps
                frames_data.append({
                    'base64': base64_image,
                    'timestamp': timestamp,
                    'formatted_time': str(timedelta(seconds=int(timestamp))),
                    'frame_number': idx
                })
        
        cap.release()
        print(f"Extracted {len(frames_data)} frames from {duration:.1f}s video")
        return frames_data
    
    def analyze_aircraft_identification(
        self,
        video_path: str,
        num_frames: int = 5
    ) -> Dict:
        """
        Identify aircraft type, airline, and registration number.
        """
        frames_data = self.extract_frames(video_path, num_frames)
        
        prompt = """Analyze these video frames and identify the aircraft.

Provide the following information:
1. Aircraft Type (e.g., Boeing 737-800, Airbus A320, etc.)
2. Airline/Operator (if visible from livery)
3. Registration Number (if visible on fuselage or tail)
4. Aircraft Color Scheme/Livery description

Respond in JSON format:
{
    "aircraft_type": "type or 'Unknown'",
    "airline": "airline name or 'Unknown'",
    "registration": "registration number or 'Not visible'",
    "livery_description": "description of aircraft appearance",
    "confidence": "high/medium/low"
}

Only include JSON in your response, no other text."""
        
        content = [{"type": "text", "text": prompt}]
        
        for frame_data in frames_data[:5]:  # Use first 5 frames for identification
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_data['base64']}"
                }
            })
        
        message = HumanMessage(content=content)
        response = self.llm.invoke([message])
        
        # FIX: Extract content from AIMessage
        response_text = response.content
        
        try:
            # Try to parse JSON response
            # Sometimes the model includes markdown code blocks
            if '```json' in response_text:
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
            elif '```' in response_text:
                json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
            
            result = json.loads(response_text)
            return result
        except Exception as e:
            print(f"Error parsing aircraft identification: {e}")
            return {
                "aircraft_type": "Unknown",
                "airline": "Unknown",
                "registration": "Not visible",
                "livery_description": "Unable to determine",
                "confidence": "low"
            }
    
    def analyze_phase_of_flight(
        self,
        video_path: str,
        num_frames: int = 10
    ) -> str:
        """Determine if video shows landing, takeoff, or taxi."""
        frames_data = self.extract_frames(video_path, num_frames)
        
        prompt = """Analyze these video frames and determine the phase of flight.

Is this video showing:
- LANDING (aircraft descending and touching down on runway)
- TAKEOFF (aircraft accelerating and lifting off runway)
- TAXI (aircraft moving on ground, taxiways, or parking)

Respond with only ONE word: LANDING, TAKEOFF, or TAXI"""
        
        content = [{"type": "text", "text": prompt}]
        
        for frame_data in frames_data:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_data['base64']}"
                }
            })
        
        message = HumanMessage(content=content)
        response = self.llm.invoke([message])
        
        # FIX: Extract content from AIMessage and parse
        response_text = response.content.strip().upper()
        
        # Validate response
        valid_phases = ["LANDING", "TAKEOFF", "TAXI"]
        
        # Check if response contains one of the valid phases
        for phase in valid_phases:
            if phase in response_text:
                return phase
        
        return "UNKNOWN"
    
    def analyze_safety_issues(
        self,
        video_path: str,
        phase_of_flight: str,
        num_frames: int = 20
    ) -> Dict:
        """
        Comprehensive safety analysis looking for damage, anomalies, and hazards.
        """
        frames_data = self.extract_frames(video_path, num_frames)
        
        prompt = f"""You are an expert aircraft safety inspector analyzing a {phase_of_flight} video.

Conduct a comprehensive safety analysis looking for:

**STRUCTURAL DAMAGE:**
- Wing damage, dents, or deformation
- Fuselage damage, cracks, or holes
- Engine cowling damage
- Landing gear damage or abnormal position
- Missing or damaged panels
- Visible fluid leaks

**EXTERNAL HAZARDS:**
- Birds or bird strikes
- Airport vehicles too close to aircraft
- Foreign object debris (FOD) on runway
- Other aircraft in unsafe proximity
- Ground equipment strikes

**OPERATIONAL ANOMALIES:**
For LANDING:
- Hard landing (excessive impact, bounce)
- High approach speed
- Wrong landing angle
- Late touchdown
- Long landing (touchdown beyond normal zone)
- Crosswind landing issues
- Brake/spoiler deployment issues

For TAKEOFF:
- Low rotation speed
- Wrong takeoff angle (too steep or too shallow)
- Abnormal acceleration
- Delayed liftoff
- Unstable climb

For TAXI:
- Excessive speed
- Erratic movements
- Clearance issues with obstacles

**CRITICAL INDICATORS:**
- Fire or smoke
- Engine shutdown/failure
- Incorrect flap configuration
- Ram Air Turbine (RAT) deployment (emergency)
- Fuel dumping
- Emergency equipment deployment

Respond in JSON format:
{{
    "damage_detected": [
        {{"type": "description", "severity": "minor/moderate/severe", "location": "area"}}
    ],
    "external_hazards": [
        {{"type": "description", "severity": "minor/moderate/severe"}}
    ],
    "operational_anomalies": [
        {{"type": "description", "severity": "minor/moderate/severe", "timestamp": "time if identifiable"}}
    ],
    "critical_indicators": [
        {{"type": "description", "severity": "critical"}}
    ],
    "normal_observations": ["list of normal/safe operations observed"],
    "overall_assessment": "detailed summary of findings"
}}

Only include JSON in your response."""
        
        content = [{"type": "text", "text": prompt}]
        
        for frame_data in frames_data:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_data['base64']}"
                }
            })
        
        message = HumanMessage(content=content)
        response = self.llm.invoke([message])
        
        # FIX: Extract content from AIMessage
        response_text = response.content
        
        try:
            # Handle markdown code blocks
            if '```json' in response_text:
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
            elif '```' in response_text:
                json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
            
            result = json.loads(response_text)
            return result
        except Exception as e:
            print(f"Error parsing safety analysis: {e}")
            print(f"Response was: {response_text[:500]}...")
            return {
                "damage_detected": [],
                "external_hazards": [],
                "operational_anomalies": [],
                "critical_indicators": [],
                "normal_observations": ["Analysis completed but format error occurred"],
                "overall_assessment": "Unable to parse detailed analysis. Manual review recommended."
            }
    
    def determine_safety_grade(self, safety_analysis: Dict) -> str:
        """
        Determine safety grade based on analysis results.
        """
        # Check for critical indicators first
        if safety_analysis.get("critical_indicators"):
            for indicator in safety_analysis["critical_indicators"]:
                critical_terms = ["fire", "smoke", "engine failure", "rat deployed", "emergency"]
                if any(term in indicator.get("type", "").lower() for term in critical_terms):
                    return "IMMEDIATE_GROUNDING"
        
        # Check for severe damage
        severe_damage = False
        for damage in safety_analysis.get("damage_detected", []):
            if damage.get("severity") == "severe":
                severe_damage = True
                break
        
        if severe_damage:
            return "IMMEDIATE_GROUNDING"
        
        # Check for moderate issues
        moderate_issues = []
        for damage in safety_analysis.get("damage_detected", []):
            if damage.get("severity") == "moderate":
                moderate_issues.append(damage)
        
        for anomaly in safety_analysis.get("operational_anomalies", []):
            if anomaly.get("severity") in ["moderate", "severe"]:
                moderate_issues.append(anomaly)
        
        if len(moderate_issues) > 0:
            # Multiple moderate issues = repair needed
            if len(moderate_issues) >= 2:
                return "REPAIR_NEEDED"
            # Single moderate issue with external hazard
            if safety_analysis.get("external_hazards"):
                return "REPAIR_NEEDED"
            return "CAUTION"
        
        # Check for minor issues
        minor_issues = (
            len(safety_analysis.get("damage_detected", [])) +
            len(safety_analysis.get("operational_anomalies", [])) +
            len(safety_analysis.get("external_hazards", []))
        )
        
        if minor_issues > 0:
            return "CAUTION"
        
        return "OK"
    
    def generate_pdf_report(
        self,
        video_path: str,
        aircraft_info: Dict,
        phase_of_flight: str,
        safety_analysis: Dict,
        safety_grade: str,
        output_filename: str
    ):
        """
        Generate comprehensive PDF report.
        """
        doc = SimpleDocTemplate(output_filename, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        story.append(Paragraph("AIRCRAFT SAFETY INSPECTION REPORT", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Report metadata
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_data = [
            ["Report Generated:", report_date],
            ["Video File:", os.path.basename(video_path)],
            ["Analysis System:", "AI-Powered Aircraft Safety Analyzer v1.0"]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Safety Grade - Large and prominent
        grade_color = self.SAFETY_GRADES[safety_grade]["color"]
        grade_desc = self.SAFETY_GRADES[safety_grade]["description"]
        
        grade_style = ParagraphStyle(
            'GradeStyle',
            parent=styles['Normal'],
            fontSize=36,
            textColor=grade_color,
            alignment=TA_CENTER,
            spaceAfter=10
        )
        
        story.append(Paragraph(f"<b>SAFETY GRADE:</b>", grade_style))
        story.append(Spacer(1, 8)) 
        story.append(Paragraph(f"<b>{safety_grade}</b>", grade_style))
        story.append(Spacer(1, 8)) 
        grade_desc_style = ParagraphStyle(
            'GradeDesc',
            parent=styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            spaceAfter=20
        )
        story.append(Spacer(1, 8)) 
        story.append(Paragraph(grade_desc, grade_desc_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Aircraft Identification
        story.append(Paragraph("1. AIRCRAFT IDENTIFICATION", heading_style))
        
        aircraft_data = [
            ["Aircraft Type:", aircraft_info.get("aircraft_type", "Unknown")],
            ["Airline/Operator:", aircraft_info.get("airline", "Unknown")],
            ["Registration:", aircraft_info.get("registration", "Not visible")],
            ["Livery:", aircraft_info.get("livery_description", "N/A")],
            ["Confidence:", aircraft_info.get("confidence", "N/A").upper()]
        ]
        
        aircraft_table = Table(aircraft_data, colWidths=[2*inch, 4*inch])
        aircraft_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(aircraft_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Phase of Flight
        story.append(Paragraph("2. PHASE OF FLIGHT", heading_style))
        story.append(Paragraph(f"<b>{phase_of_flight}</b>", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Overall Assessment
        story.append(Paragraph("3. OVERALL ASSESSMENT", heading_style))
        assessment_text = safety_analysis.get("overall_assessment", "No assessment available")
        story.append(Paragraph(assessment_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Damage Detected
        story.append(Paragraph("4. STRUCTURAL DAMAGE ANALYSIS", heading_style))
        
        if safety_analysis.get("damage_detected"):
            damage_data = [["Type", "Severity", "Location"]]
            for damage in safety_analysis["damage_detected"]:
                damage_data.append([
                    damage.get("type", "N/A"),
                    damage.get("severity", "N/A").upper(),
                    damage.get("location", "N/A")
                ])
            
            damage_table = Table(damage_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
            damage_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(damage_table)
        else:
            story.append(Paragraph("✓ No structural damage detected", styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # External Hazards
        story.append(Paragraph("5. EXTERNAL HAZARDS", heading_style))
        
        if safety_analysis.get("external_hazards"):
            for i, hazard in enumerate(safety_analysis["external_hazards"], 1):
                hazard_text = f"{i}. {hazard.get('type', 'N/A')} - Severity: {hazard.get('severity', 'N/A').upper()}"
                story.append(Paragraph(hazard_text, styles['Normal']))
        else:
            story.append(Paragraph("✓ No external hazards detected", styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Operational Anomalies
        story.append(Paragraph("6. OPERATIONAL ANOMALIES", heading_style))
        
        if safety_analysis.get("operational_anomalies"):
            for i, anomaly in enumerate(safety_analysis["operational_anomalies"], 1):
                anomaly_text = f"{i}. {anomaly.get('type', 'N/A')} - Severity: {anomaly.get('severity', 'N/A').upper()}"
                if anomaly.get("timestamp"):
                    anomaly_text += f" (Time: {anomaly['timestamp']})"
                story.append(Paragraph(anomaly_text, styles['Normal']))
        else:
            story.append(Paragraph("✓ No operational anomalies detected", styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Critical Indicators
        if safety_analysis.get("critical_indicators"):
            story.append(Paragraph("7. ⚠️ CRITICAL INDICATORS", heading_style))
            
            for i, indicator in enumerate(safety_analysis["critical_indicators"], 1):
                critical_style = ParagraphStyle(
                    'Critical',
                    parent=styles['Normal'],
                    textColor=colors.red,
                    fontSize=11
                )
                indicator_text = f"<b>{i}. {indicator.get('type', 'N/A')}</b>"
                story.append(Paragraph(indicator_text, critical_style))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Normal Observations
        story.append(Paragraph("8. NORMAL OPERATIONS OBSERVED", heading_style))
        
        if safety_analysis.get("normal_observations"):
            for i, observation in enumerate(safety_analysis["normal_observations"], 1):
                story.append(Paragraph(f"• {observation}", styles['Normal']))
        else:
            story.append(Paragraph("No normal operations documented", styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        story.append(Paragraph("9. RECOMMENDATIONS", heading_style))
        
        recommendations = self._generate_recommendations(safety_grade, safety_analysis)
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("--- END OF REPORT ---", footer_style))
        story.append(Paragraph(
            "This report was generated by AI analysis and should be verified by qualified aviation personnel.",
            footer_style
        ))
        
        # Build PDF
        doc.build(story)
        print(f"\n✓ PDF report generated: {output_filename}")
    
    def _generate_recommendations(self, safety_grade: str, safety_analysis: Dict) -> List[str]:
        """Generate recommendations based on safety grade and analysis."""
        recommendations = []
        
        if safety_grade == "IMMEDIATE_GROUNDING":
            recommendations.append("GROUND AIRCRAFT IMMEDIATELY - Do not operate until issues are resolved")
            recommendations.append("Conduct comprehensive safety inspection by certified personnel")
            recommendations.append("Document all findings and submit incident report to aviation authority")
            recommendations.append("Review flight data recorder and cockpit voice recorder if applicable")
        
        elif safety_grade == "REPAIR_NEEDED":
            recommendations.append("Aircraft should not fly until repairs are completed and verified")
            recommendations.append("Conduct detailed inspection of identified damage areas")
            recommendations.append("Complete all necessary repairs per manufacturer specifications")
            recommendations.append("Perform test flight and re-inspection before returning to service")
        
        elif safety_grade == "CAUTION":
            recommendations.append("Conduct pre-flight inspection focusing on identified areas of concern")
            recommendations.append("Monitor aircraft systems closely on next flight")
            recommendations.append("Schedule detailed inspection at next maintenance opportunity")
            recommendations.append("Document all observations in aircraft maintenance log")
        
        else:  # OK
            recommendations.append("Aircraft appears safe for normal operations")
            recommendations.append("Continue routine maintenance schedule")
            recommendations.append("Monitor for any changes in subsequent operations")
        
        # Add specific recommendations based on issues found
        if safety_analysis.get("damage_detected"):
            recommendations.append("Inspect and repair all detected structural damage")
        
        if safety_analysis.get("operational_anomalies"):
            recommendations.append("Investigate operational anomalies and verify system functionality")
        
        return recommendations
    
    def analyze_video_and_generate_report(
        self,
        video_path: str,
        output_pdf: Optional[str] = None
    ) -> str:
        """
        Complete analysis pipeline: analyze video and generate PDF report.
        
        Args:
            video_path: Path to the video file
            output_pdf: Path for output PDF (auto-generated if not provided)
            
        Returns:
            Path to generated PDF report
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Generate output filename if not provided
        if not output_pdf:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_pdf = f"Aircraft_Safety_Report_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        print("\n" + "="*70)
        print("AIRCRAFT SAFETY ANALYSIS SYSTEM")
        print("="*70)
        print(f"\nAnalyzing video: {video_path}\n")
        
        # Step 1: Identify aircraft
        print("Step 1/4: Identifying aircraft...")
        aircraft_info = self.analyze_aircraft_identification(video_path, num_frames=5)
        print(f"  → Aircraft: {aircraft_info.get('aircraft_type', 'Unknown')}")
        print(f"  → Airline: {aircraft_info.get('airline', 'Unknown')}")
        
        # Step 2: Determine phase of flight
        print("\nStep 2/4: Determining phase of flight...")
        phase_of_flight = self.analyze_phase_of_flight(video_path, num_frames=10)
        print(f"  → Phase: {phase_of_flight}")
        
        # Step 3: Safety analysis
        print("\nStep 3/4: Conducting comprehensive safety analysis...")
        print("  (This may take 30-60 seconds...)")
        safety_analysis = self.analyze_safety_issues(video_path, phase_of_flight, num_frames=20)
        
        # Step 4: Determine safety grade
        print("\nStep 4/4: Determining safety grade...")
        safety_grade = self.determine_safety_grade(safety_analysis)
        print(f"  → Safety Grade: {safety_grade}")
        
        # Generate PDF report
        print(f"\nGenerating PDF report: {output_pdf}")
        self.generate_pdf_report(
            video_path,
            aircraft_info,
            phase_of_flight,
            safety_analysis,
            safety_grade,
            output_pdf
        )
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nSafety Grade: {safety_grade}")
        print(f"Report saved: {output_pdf}")
        
        return output_pdf


# Example usage
if __name__ == "__main__":
    analyzer = AircraftSafetyAnalyzer()
    
    # Update this with your video path
    video_path = "normal-takeoff.mp4"
    
    if not os.path.exists(video_path):
        print("❌ Video file not found!")
        print(f"   Looking for: {video_path}")
        print("\nPlease update the video_path variable with your aircraft video file.")
    else:
        # Run complete analysis
        report_path = analyzer.analyze_video_and_generate_report(
            video_path=video_path,
            output_pdf="aircraft_safety_report.pdf"
        )
        
        print(f"\n✅ Report ready: {report_path}")