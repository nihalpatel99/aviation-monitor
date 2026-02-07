"""
Aircraft Safety Analysis - Streamlit Web Application
Production-ready version with enhanced UI/UX and error handling
"""

import streamlit as st  
import os
from dotenv import load_dotenv
import base64
from typing import List, Dict, Optional
import tempfile
import cv2
from datetime import timedelta, datetime
import json
import re
import traceback

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

# Load environment variables
load_dotenv()


class AircraftSafetyAnalyzer:
    """Analyze aircraft videos for safety issues and generate reports."""
    
    # Safety grading system
    SAFETY_GRADES = {
        "OK": {
            "color": colors.green,
            "description": "No issues detected - aircraft safe for operation",
            "icon": "✅"
        },
        "CAUTION": {
            "color": colors.yellow,
            "description": "Minor issues requiring inspection before next flight",
            "icon": "⚠️"
        },
        "REPAIR_NEEDED": {
            "color": colors.orange,
            "description": "Damage or issues requiring repair before next operation",
            "icon": "🔧"
        },
        "IMMEDIATE_GROUNDING": {
            "color": colors.red,
            "description": "Critical safety issue - aircraft must be grounded immediately",
            "icon": "🛑"
        }
    }
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize the aircraft safety analyzer."""
        if not api_key:
            raise ValueError("API key is required")
            
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            max_tokens=4096,
            temperature=0.3
        )
        self.output_parser = StrOutputParser()
    
    def extract_frames(
        self,
        video_path: str,
        num_frames: int = 20,
        progress_callback=None
    ) -> List[Dict]:
        """Extract frames from video with progress tracking."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Unable to open video file. Please check the file format.")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        if total_frames == 0:
            raise ValueError("Video appears to be empty or corrupted")
        
        frames_data = []
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
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
                
                if progress_callback:
                    progress_callback(i + 1, num_frames)
        
        cap.release()
        return frames_data
    
    def analyze_aircraft_identification(
        self,
        video_path: str,
        num_frames: int = 5,
        progress_callback=None
    ) -> Dict:
        """Identify aircraft type, airline, and registration number."""
        frames_data = self.extract_frames(video_path, num_frames, progress_callback)
        
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
        
        for frame_data in frames_data[:5]:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_data['base64']}"
                }
            })
        
        message = HumanMessage(content=content)
        response = self.llm.invoke([message])
        response_text = response.content
        
        try:
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
        num_frames: int = 10,
        progress_callback=None
    ) -> str:
        """Determine if video shows landing, takeoff, or taxi."""
        frames_data = self.extract_frames(video_path, num_frames, progress_callback)
        
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
        response_text = response.content.strip().upper()
        
        valid_phases = ["LANDING", "TAKEOFF", "TAXI"]
        
        for phase in valid_phases:
            if phase in response_text:
                return phase
        
        return "UNKNOWN"
    
    def analyze_safety_issues(
        self,
        video_path: str,
        phase_of_flight: str,
        num_frames: int = 20,
        progress_callback=None
    ) -> Dict:
        """Comprehensive safety analysis."""
        frames_data = self.extract_frames(video_path, num_frames, progress_callback)
        
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
        response_text = response.content
        
        try:
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
            return {
                "damage_detected": [],
                "external_hazards": [],
                "operational_anomalies": [],
                "critical_indicators": [],
                "normal_observations": ["Analysis completed but format error occurred"],
                "overall_assessment": "Unable to parse detailed analysis. Manual review recommended."
            }
    
    def determine_safety_grade(self, safety_analysis: Dict) -> str:
        """Determine safety grade based on analysis results."""
        if safety_analysis.get("critical_indicators"):
            for indicator in safety_analysis["critical_indicators"]:
                critical_terms = ["fire", "smoke", "engine failure", "rat deployed", "emergency"]
                if any(term in indicator.get("type", "").lower() for term in critical_terms):
                    return "IMMEDIATE_GROUNDING"
        
        severe_damage = False
        for damage in safety_analysis.get("damage_detected", []):
            if damage.get("severity") == "severe":
                severe_damage = True
                break
        
        if severe_damage:
            return "IMMEDIATE_GROUNDING"
        
        moderate_issues = []
        for damage in safety_analysis.get("damage_detected", []):
            if damage.get("severity") == "moderate":
                moderate_issues.append(damage)
        
        for anomaly in safety_analysis.get("operational_anomalies", []):
            if anomaly.get("severity") in ["moderate", "severe"]:
                moderate_issues.append(anomaly)
        
        if len(moderate_issues) > 0:
            if len(moderate_issues) >= 2:
                return "REPAIR_NEEDED"
            if safety_analysis.get("external_hazards"):
                return "REPAIR_NEEDED"
            return "CAUTION"
        
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
        video_name: str,
        aircraft_info: Dict,
        phase_of_flight: str,
        safety_analysis: Dict,
        safety_grade: str,
        output_filename: str
    ):
        """Generate comprehensive PDF report."""
        doc = SimpleDocTemplate(output_filename, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
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
        
        story.append(Paragraph("AIRCRAFT SAFETY INSPECTION REPORT", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_data = [
            ["Report Generated:", report_date],
            ["Video File:", video_name],
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
        
        story.append(Paragraph("2. PHASE OF FLIGHT", heading_style))
        story.append(Paragraph(f"<b>{phase_of_flight}</b>", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("3. OVERALL ASSESSMENT", heading_style))
        assessment_text = safety_analysis.get("overall_assessment", "No assessment available")
        story.append(Paragraph(assessment_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
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
        
        story.append(Paragraph("5. EXTERNAL HAZARDS", heading_style))
        
        if safety_analysis.get("external_hazards"):
            for i, hazard in enumerate(safety_analysis["external_hazards"], 1):
                hazard_text = f"{i}. {hazard.get('type', 'N/A')} - Severity: {hazard.get('severity', 'N/A').upper()}"
                story.append(Paragraph(hazard_text, styles['Normal']))
        else:
            story.append(Paragraph("✓ No external hazards detected", styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        
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
        
        story.append(Paragraph("8. NORMAL OPERATIONS OBSERVED", heading_style))
        
        if safety_analysis.get("normal_observations"):
            for i, observation in enumerate(safety_analysis["normal_observations"], 1):
                story.append(Paragraph(f"• {observation}", styles['Normal']))
        else:
            story.append(Paragraph("No normal operations documented", styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("9. RECOMMENDATIONS", heading_style))
        
        recommendations = self._generate_recommendations(safety_grade, safety_analysis)
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
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
        
        doc.build(story)
    
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
        
        else:
            recommendations.append("Aircraft appears safe for normal operations")
            recommendations.append("Continue routine maintenance schedule")
            recommendations.append("Monitor for any changes in subsequent operations")
        
        if safety_analysis.get("damage_detected"):
            recommendations.append("Inspect and repair all detected structural damage")
        
        if safety_analysis.get("operational_anomalies"):
            recommendations.append("Investigate operational anomalies and verify system functionality")
        
        return recommendations


# Streamlit UI
def main():
    st.set_page_config(
        page_title="luftfartsovervåker",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .safety-grade {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .grade-ok { background-color: #d4edda; color: #155724; }
        .grade-caution { background-color: #fff3cd; color: #856404; }
        .grade-repair { background-color: #f8d7da; color: #721c24; }
        .grade-grounding { background-color: #f5c6cb; color: #721c24; }
        .info-box {
            background-color: #d93232;
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid #1f77b4;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid: #ffc107;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    col11, col12, col13 = st.columns([1, 1, 1])

    with col12:
        st.image("luftfartsovervåker.png", width=150)
    
    st.markdown('<div class="main-header">✈️ luftfartsovervåker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Aviation Safety Analysis System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your API key from https://platform.openai.com/account/api-keys",
            placeholder="sk-..."
        )
        
        st.markdown("---")
        
        # Model selection
        model = st.selectbox(
            "Model",
            ["gpt-4o", "gpt-4-turbo"],
            index=0,
            help="GPT-4o is faster and cheaper (recommended)"
        )
        
        # Frame count
        frame_count = st.slider(
            "Analysis Detail",
            min_value=10,
            max_value=30,
            value=20,
            help="More frames = more accurate but slower and more expensive"
        )
        
        st.markdown("---")
        
        # Info section
        st.markdown("### 📊 What We Detect")
        st.markdown("""
        - **Structural Damage**: Wing, fuselage, engine issues
        - **External Hazards**: Birds, vehicles, FOD
        - **Operational Issues**: Hard landings, wrong angles
        - **Critical Events**: Fire, smoke, engine failure
        """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Safety Grades")
        st.markdown("""
        - ✅ **OK**: Safe for operation
        - ⚠️ **CAUTION**: Inspection needed
        - 🔧 **REPAIR NEEDED**: Ground until repair
        - 🛑 **IMMEDIATE GROUNDING**: Critical issue
        """)
    
    # Main content
    if not api_key:
        st.markdown('<div class="info-box">👈 Please enter your OpenAI API key in the sidebar to begin</div>', unsafe_allow_html=True)
        
        st.markdown("### 🚀 How It Works")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 1️⃣ Upload Video")
            st.write("Upload a video of aircraft landing, takeoff, or taxi operations")
        
        with col2:
            st.markdown("#### 2️⃣ AI Analysis")
            st.write("Our AI analyzes frames for safety issues and damage")
        
        with col3:
            st.markdown("#### 3️⃣ Get Report")
            st.write("Download comprehensive PDF safety report")
        
        st.markdown("---")
        
        st.markdown("### ✨ Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - Aircraft identification (type, airline, registration)
            - Phase of flight detection (landing/takeoff/taxi)
            - Structural damage analysis
            - Operational anomaly detection
            """)
        
        with col2:
            st.markdown("""
            - External hazard identification
            - Critical indicator monitoring
            - Automated safety grading
            - Professional PDF reports
            """)
        
        st.stop()
    
    # File uploader
    st.markdown("### 📹 Upload Aircraft Video")
    video_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv"],
        help="Supported formats: MP4, MOV, AVI, MKV"
    )
    
    if video_file:
        # Display video info
        file_size_mb = len(video_file.getvalue()) / (1024 * 1024)
        st.info(f"📁 **File:** {video_file.name} | **Size:** {file_size_mb:.2f} MB")
        
        # Show video preview
        with st.expander("🎬 Video Preview"):
            st.video(video_file)
        
        # Analysis button
        if st.button("🔍 Analyze Aircraft Safety", type="primary", use_container_width=True):
            try:
                # Initialize analyzer
                with st.spinner("Initializing AI analyzer..."):
                    analyzer = AircraftSafetyAnalyzer(api_key=api_key, model=model)
                
                # Save uploaded video to temp file
                suffix = os.path.splitext(video_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                    tfile.write(video_file.read())
                    video_path = tfile.name
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Identify aircraft
                status_text.text("🔍 Step 1/4: Identifying aircraft...")
                progress_bar.progress(10)
                
                def update_progress_1(current, total):
                    progress_bar.progress(10 + int(15 * current / total))
                
                aircraft_info = analyzer.analyze_aircraft_identification(
                    video_path, 
                    num_frames=5,
                    progress_callback=update_progress_1
                )
                
                # Display aircraft info
                st.success(f"✓ Aircraft: {aircraft_info.get('aircraft_type', 'Unknown')}")
                
                # Step 2: Determine phase
                status_text.text("✈️ Step 2/4: Determining phase of flight...")
                progress_bar.progress(30)
                
                def update_progress_2(current, total):
                    progress_bar.progress(30 + int(15 * current / total))
                
                phase_of_flight = analyzer.analyze_phase_of_flight(
                    video_path,
                    num_frames=10,
                    progress_callback=update_progress_2
                )
                
                st.success(f"✓ Phase: {phase_of_flight}")
                
                # Step 3: Safety analysis
                status_text.text("🔬 Step 3/4: Conducting comprehensive safety analysis...")
                progress_bar.progress(50)
                
                def update_progress_3(current, total):
                    progress_bar.progress(50 + int(30 * current / total))
                
                safety_analysis = analyzer.analyze_safety_issues(
                    video_path,
                    phase_of_flight,
                    num_frames=frame_count,
                    progress_callback=update_progress_3
                )
                
                # Step 4: Determine grade
                status_text.text("📊 Step 4/4: Determining safety grade...")
                progress_bar.progress(85)
                
                safety_grade = analyzer.determine_safety_grade(safety_analysis)
                
                # Generate PDF
                status_text.text("📄 Generating PDF report...")
                progress_bar.progress(90)
                
                base_name = video_file.name.rsplit(".", 1)[0]
                output_pdf = f"Aircraft_Safety_Report_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                
                analyzer.generate_pdf_report(
                    video_file.name,
                    aircraft_info,
                    phase_of_flight,
                    safety_analysis,
                    safety_grade,
                    output_pdf
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Clean up temp file
                try:
                    os.unlink(video_path)
                except:
                    pass
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Safety Grade (large display)
                grade_info = analyzer.SAFETY_GRADES[safety_grade]
                grade_classes = {
                    "OK": "grade-ok",
                    "CAUTION": "grade-caution",
                    "REPAIR_NEEDED": "grade-repair",
                    "IMMEDIATE_GROUNDING": "grade-grounding"
                }
                
                st.markdown(
                    f'<div class="safety-grade {grade_classes[safety_grade]}">'
                    f'{grade_info["icon"]} SAFETY GRADE: {safety_grade}<br>'
                    f'<small style="font-size:1rem;">{grade_info["description"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Detailed results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🛩️ Aircraft Information")
                    st.write(f"**Type:** {aircraft_info.get('aircraft_type', 'Unknown')}")
                    st.write(f"**Airline:** {aircraft_info.get('airline', 'Unknown')}")
                    st.write(f"**Registration:** {aircraft_info.get('registration', 'Not visible')}")
                    st.write(f"**Phase:** {phase_of_flight}")
                
                with col2:
                    st.markdown("### 📋 Issues Detected")
                    st.write(f"**Structural Damage:** {len(safety_analysis.get('damage_detected', []))}")
                    st.write(f"**External Hazards:** {len(safety_analysis.get('external_hazards', []))}")
                    st.write(f"**Operational Anomalies:** {len(safety_analysis.get('operational_anomalies', []))}")
                    st.write(f"**Critical Indicators:** {len(safety_analysis.get('critical_indicators', []))}")
                
                # Overall Assessment
                with st.expander("📝 Overall Assessment", expanded=True):
                    st.write(safety_analysis.get('overall_assessment', 'No assessment available'))
                
                # Detailed findings
                if safety_analysis.get('damage_detected'):
                    with st.expander("⚠️ Structural Damage Details"):
                        for i, damage in enumerate(safety_analysis['damage_detected'], 1):
                            st.warning(f"**{i}. {damage.get('type', 'N/A')}**\n- Severity: {damage.get('severity', 'N/A').upper()}\n- Location: {damage.get('location', 'N/A')}")
                
                if safety_analysis.get('operational_anomalies'):
                    with st.expander("🔍 Operational Anomalies"):
                        for i, anomaly in enumerate(safety_analysis['operational_anomalies'], 1):
                            st.warning(f"**{i}. {anomaly.get('type', 'N/A')}**\n- Severity: {anomaly.get('severity', 'N/A').upper()}")
                
                if safety_analysis.get('critical_indicators'):
                    with st.expander("🚨 Critical Indicators", expanded=True):
                        for i, indicator in enumerate(safety_analysis['critical_indicators'], 1):
                            st.error(f"**{i}. {indicator.get('type', 'N/A')}**")
                
                # Download report
                st.markdown("---")
                st.markdown("### 📥 Download Report")
                
                with open(output_pdf, "rb") as f:
                    st.download_button(
                        label="📄 Download PDF Safety Report",
                        data=f.read(),
                        file_name=output_pdf,
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary"
                    )
                
                # Clean up PDF file
                try:
                    os.unlink(output_pdf)
                except:
                    pass
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)
                
                # Clean up on error
                try:
                    if 'video_path' in locals():
                        os.unlink(video_path)
                    if 'output_pdf' in locals() and os.path.exists(output_pdf):
                        os.unlink(output_pdf)
                except:
                    pass
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        ⚠️ <strong>Disclaimer:</strong> This tool uses AI and should not replace qualified aviation personnel. 
        All findings must be verified by certified inspectors.
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()