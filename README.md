# ✈️ Aircraft Safety Monitor - Streamlit Web App

AI-powered aircraft safety analysis system that detects damage, hazards, and operational anomalies in landing, takeoff, and taxi videos.

## 🌟 Live Demo

**Deploy to Streamlit Cloud:** [[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://your-app-url.streamlit.app]

## 🚀 Features

### Detection Capabilities
- **Aircraft Identification**: Type, airline, registration number
- **Phase Detection**: Landing, takeoff, or taxi
- **Structural Damage**: Wings, fuselage, engines, landing gear
- **External Hazards**: Birds, vehicles, FOD, proximity issues
- **Operational Anomalies**: Speed, angle, configuration issues
- **Critical Indicators**: Fire, smoke, engine failure, emergency systems

### Outputs
- **Automated Safety Grading**: OK → Caution → Repair Needed → Immediate Grounding
- **Professional PDF Reports**: Comprehensive safety reports
- **Real-time Analysis**: Progress tracking and status updates
- **Interactive UI**: Video preview and detailed findings

## 📦 Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/aircraft-safety-monitor.git
cd aircraft-safety-monitor

# Install dependencies
pip install -r requirements_streamlit.txt

# Run the app
streamlit run app.py
```

### Environment Variables

Create a `.env` file (optional - can also use UI):
```
OPENAI_API_KEY=your-api-key-here
```

## 🌐 Deploy to Streamlit Cloud

### Option 1: One-Click Deploy

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your forked repository
5. Set main file path: `app.py`
6. Click "Deploy"

### Option 2: Manual Setup

1. **Prepare Your Repository**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Repository: `yourusername/aircraft-safety-monitor`
   - Branch: `main`
   - Main file path: `app.py`

3. **Set Secrets (Optional)**
   - Go to app settings → Secrets
   - Add your OpenAI API key:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

## 🔧 Configuration

### Streamlit Configuration

The app includes a `.streamlit/config.toml` file with:
- Custom theme colors
- Max upload size: 200MB
- Security settings
- Browser preferences

### Deployment Settings

For production deployment:

1. **requirements_streamlit.txt**: All dependencies
2. **.streamlit/config.toml**: App configuration
3. **Secrets Management**: Use Streamlit Cloud secrets for API keys

## 💡 Usage

1. **Enter API Key**: Add your OpenAI API key in the sidebar
2. **Upload Video**: Choose an aircraft video (MP4, MOV, AVI, MKV)
3. **Configure Analysis**: Select model and detail level
4. **Analyze**: Click "Analyze Aircraft Safety"
5. **Download Report**: Get comprehensive PDF safety report

## 📊 Safety Grading System

| Grade | Icon | Description |
|-------|------|-------------|
| **OK** | ✅ | No issues detected - safe for operation |
| **CAUTION** | ⚠️ | Minor issues requiring inspection |
| **REPAIR NEEDED** | 🔧 | Ground until repairs completed |
| **IMMEDIATE GROUNDING** | 🛑 | Critical safety issue detected |

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **AI/LLM**: OpenAI GPT-4o Vision
- **Video Processing**: OpenCV
- **PDF Generation**: ReportLab
- **Framework**: LangChain

## 📁 Project Structure

```
aircraft-safety-monitor/
├── app.py                          # Main Streamlit application
├── requirements_streamlit.txt      # Python dependencies
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
├── README.md                       # This file
└── .env                            # Environment variables (local only)
```

## ⚙️ Advanced Configuration

### Model Selection

The app supports:
- **GPT-4o** (Recommended): Faster and cheaper
- **GPT-4 Turbo**: More capable for complex scenarios

### Analysis Detail

Adjust frame count (10-30):
- **10 frames**: Quick analysis (~$0.02)
- **20 frames**: Standard analysis (~$0.04)
- **30 frames**: Detailed analysis (~$0.06)

## 🔐 Security Best Practices

### For Deployment:

1. **Never commit API keys** to the repository
2. **Use Streamlit Secrets** for production API keys
3. **Enable XSRF protection** (already configured)
4. **Limit file upload size** (configured to 200MB)
5. **Validate all inputs** (implemented)

### API Key Management:

**Development:**
```bash
# Use .env file
OPENAI_API_KEY=sk-...
```

**Production:**
```toml
# Streamlit Cloud Secrets
OPENAI_API_KEY = "sk-..."
```

## 🐛 Troubleshooting

### Common Issues

**1. "opencv-python" not found**
- Solution: Use `opencv-python-headless` in requirements (already configured)

**2. File upload fails**
- Check file size < 200MB
- Ensure supported format (MP4, MOV, AVI, MKV)

**3. API errors**
- Verify API key is correct
- Check OpenAI account has credits
- Ensure internet connection is stable

**4. PDF generation fails**
- Check reportlab is installed
- Verify write permissions

## 💰 Cost Estimation

Based on OpenAI GPT-4o pricing:

| Frames | Cost per Video |
|--------|---------------|
| 10 | ~$0.02-0.03 |
| 20 | ~$0.04-0.06 |
| 30 | ~$0.06-0.09 |

## ⚠️ Important Disclaimers

1. **AI-Assisted Tool**: Should not replace qualified aviation personnel
2. **Verification Required**: All findings must be verified by certified inspectors
3. **Not Certified**: Not certified for official aviation safety reporting
4. **Educational Purpose**: Use as supplementary safety tool only

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues:
1. Check the troubleshooting section
2. Review Streamlit Cloud logs
3. Open an issue on GitHub

## 🔄 Updates & Changelog

### Version 1.0.0 (Current)
- Initial release
- GPT-4o Vision integration
- PDF report generation
- Streamlit Cloud deployment ready
- Comprehensive error handling
- Progress tracking

## 🌟 Acknowledgments

- OpenAI for GPT-4 Vision API
- Streamlit for the amazing framework
- LangChain for LLM orchestration
- Aviation safety professionals for domain expertise

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [LangChain Documentation](https://python.langchain.com)
- [Aviation Safety Guidelines](https://www.faa.gov)

---

**Made with ❤️ for aviation safety**


