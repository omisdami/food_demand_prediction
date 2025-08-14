# 🍗 Restaurant Inventory Forecasting Dashboard

A comprehensive web-based dashboard for restaurant inventory forecasting using machine learning models including regression, ARIMA time series, and anomaly detection. This production-ready system provides managers with an intuitive interface to upload data, generate forecasts, and make data-driven inventory decisions.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-2.3+-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 🎯 Features

### 📊 **Advanced Forecasting Models**
- **Regression Models**: Lasso, Ridge, Linear, ElasticNet, Random Forest (87.1% accuracy)
- **ARIMA Time Series**: Seasonal and non-seasonal models with external regressors
- **Model Comparison**: Automatic selection of best-performing model
- **Anomaly Detection**: AI-powered detection of unusual inventory patterns

### 🖥️ **Manager-Friendly Interface**
- **Drag & Drop Upload**: Easy CSV file upload with validation
- **Interactive Charts**: Chart.js visualizations with item selection
- **Real-time Results**: Dynamic updates without page refresh
- **Export Functionality**: CSV export of forecast results
- **Responsive Design**: Works on desktop, tablet, and mobile

### 🚀 **Production Ready**
- **Docker Containerization**: Complete containerized deployment
- **Health Monitoring**: Built-in health checks and status monitoring
- **Error Handling**: Comprehensive error handling and user feedback
- **Performance Optimized**: Efficient model loading and caching
- **Security**: File validation and secure upload handling

## 📋 Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose** (for containerized deployment)
- **8GB RAM minimum** (for machine learning models)
- **2GB disk space** (for models and data)

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository**
```bash
git clone <repository-url>
cd restaurant-inventory-forecasting
```

2. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

3. **Access the application**
- Open your browser to `http://localhost:5000`
- The dashboard will be ready to use immediately

### Option 2: Local Development Setup

1. **Clone and setup environment**
```bash
git clone <repository-url>
cd restaurant-inventory-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Run the application**
```bash
python app.py
```

3. **Access the application**
- Open your browser to `http://localhost:5000`

## 📁 Project Structure

```
restaurant-inventory-forecasting/
├── 🐳 Docker Configuration
│   ├── Dockerfile                 # Container definition
│   ├── docker-compose.yml         # Multi-service orchestration
│   ├── nginx.conf                 # Production reverse proxy
│   └── requirements.txt           # Python dependencies
│
├── 🌐 Web Application
│   ├── app.py                     # Flask backend API
│   ├── templates/
│   │   └── index.html             # Main dashboard UI
│   └── static/
│       ├── css/style.css          # Modern responsive styling
│       └── js/app.js              # Interactive frontend logic
│
├── 🤖 ML Models & Logic
│   ├── inventory_forecasting_regression.py  # Regression models
│   ├── arima_forecasting.py                # ARIMA time series
│   ├── restaurant_forecast_tool.py         # CLI tool
│   └── Autoencoder/                        # Anomaly detection
│       ├── autoencoder_anomaly_detection.py
│       └── Autoencoder_Documentation.md
│
├── 📊 Data & Results
│   ├── data/                      # Sample datasets
│   ├── models/                    # Trained model storage
│   ├── results/                   # Analysis results
│   ├── forecasts/                 # Generated forecasts
│   └── uploads/                   # User uploaded files
│
└── 📚 Documentation
    ├── README.md                  # This file
    ├── model_improvement_journey.md
    └── Autoencoder/Autoencoder_Documentation.md
```

## 🎮 How to Use

### 1. **Upload Your Data**
- Drag and drop a CSV file or click to browse
- Required columns: `delivery_date`, `wings`, `tenders`, `fries_reg`, `fries_large`, `veggies`, `dips`, `drinks`, `flavours`
- File validation ensures data integrity

### 2. **Configure Forecast**
- **Model Type**: Choose Regression (recommended), ARIMA, or Both
- **Forecast Days**: Select 7, 14, 21, or 30 days
- **Anomaly Detection**: Enable to detect unusual patterns

### 3. **Generate Forecast**
- Click "Generate Forecast" to start processing
- Real-time progress updates during model training
- Results appear automatically when complete

### 4. **Analyze Results**
- **Model Performance**: View accuracy metrics and model comparison
- **Anomaly Detection**: See detected unusual patterns
- **Summary Statistics**: Weekly totals and insights
- **Interactive Chart**: Visualize forecasts with Chart.js
- **Detailed Table**: Complete forecast data with weekend highlighting

### 5. **Export Results**
- Click "Export CSV" to download forecast data
- Includes all forecast values and recommended stock levels

## 📊 Sample Data Format

Your CSV file should have this structure:

```csv
delivery_date,wings,tenders,fries_reg,fries_large,veggies,dips,drinks,flavours
2024-01-01,5139,545,131,145,140,471,217,721
2024-01-06,5225,577,117,160,157,475,175,718
2024-01-08,4682,623,157,137,132,470,237,735
...
```

**Column Descriptions:**
- `delivery_date`: Date of inventory delivery (YYYY-MM-DD)
- `wings`: Number of chicken wings
- `tenders`: Number of chicken tenders
- `fries_reg`: Regular fries portions
- `fries_large`: Large fries portions
- `veggies`: Veggie stick portions
- `dips`: Dip containers
- `drinks`: Fountain drinks
- `flavours`: Sauce/flavor servings

## 🐳 Docker Deployment

### Development Environment
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Environment with Nginx
```bash
# Run with production profile (includes Nginx reverse proxy)
docker-compose --profile production up -d

# Access via http://localhost (port 80)
```

### Individual Container
```bash
# Build image
docker build -t restaurant-inventory-app .

# Run container
docker run -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/models:/app/models \
  restaurant-inventory-app
```

## ☁️ Cloud Deployment

### AWS Deployment

1. **EC2 Instance Setup**
```bash
# Launch EC2 instance (t3.medium or larger recommended)
# Install Docker and Docker Compose
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

2. **Deploy Application**
```bash
# Clone repository
git clone <your-repo-url>
cd restaurant-inventory-forecasting

# Run with production profile
docker-compose --profile production up -d
```

3. **Configure Security Group**
- Allow inbound traffic on port 80 (HTTP)
- Allow inbound traffic on port 443 (HTTPS) if using SSL

### Azure Container Instances

1. **Create Resource Group**
```bash
az group create --name restaurant-inventory-rg --location eastus
```

2. **Deploy Container**
```bash
az container create \
  --resource-group restaurant-inventory-rg \
  --name restaurant-inventory-app \
  --image <your-docker-registry>/restaurant-inventory-app:latest \
  --ports 5000 \
  --memory 4 \
  --cpu 2
```

### Render.com Deployment

1. **Create `render.yaml`**
```yaml
services:
  - type: web
    name: restaurant-inventory-app
    env: docker
    dockerfilePath: ./Dockerfile
    plan: standard
    envVars:
      - key: FLASK_ENV
        value: production
```

2. **Connect Repository**
- Connect your GitHub repository to Render
- Automatic deployments on push to main branch

### Heroku Deployment

1. **Create `heroku.yml`**
```yaml
build:
  docker:
    web: Dockerfile
run:
  web: python app.py
```

2. **Deploy**
```bash
heroku create restaurant-inventory-app
heroku stack:set container
git push heroku main
```

## 🔧 Configuration

### Environment Variables

```bash
# Flask Configuration
FLASK_ENV=production
FLASK_APP=app.py

# Application Settings
MAX_CONTENT_LENGTH=16777216  # 16MB file upload limit
UPLOAD_FOLDER=uploads

# Model Settings
MODEL_CACHE_DIR=models
RESULTS_DIR=results
```

### Docker Environment Variables

```yaml
# In docker-compose.yml
environment:
  - FLASK_ENV=production
  - PYTHONPATH=/app
  - MAX_WORKERS=4
```

## 📈 Performance Optimization

### Model Performance
- **Regression Models**: 87.1% accuracy (Lasso best performer)
- **ARIMA Models**: Suitable for time series patterns
- **Anomaly Detection**: Real-time unusual pattern detection

### System Performance
- **Memory Usage**: ~2-4GB during model training
- **CPU Usage**: Multi-core utilization during training
- **Storage**: Models cached for faster subsequent runs
- **Response Time**: <30 seconds for forecast generation

### Scaling Recommendations
- **CPU**: 2+ cores recommended for production
- **Memory**: 8GB+ for large datasets and multiple models
- **Storage**: SSD recommended for model loading performance
- **Network**: Consider CDN for static assets in production

## 🛠️ API Endpoints

### File Upload
```http
POST /api/upload
Content-Type: multipart/form-data

Response: {
  "success": true,
  "filename": "uploaded_file.csv",
  "rows": 106,
  "columns": ["delivery_date", "wings", ...]
}
```

### Generate Forecast
```http
POST /api/forecast
Content-Type: application/json

{
  "filepath": "/path/to/file.csv",
  "model_type": "regression",
  "forecast_days": 7,
  "anomaly_detection": true
}

Response: {
  "success": true,
  "forecast_data": [...],
  "model_performance": {...},
  "anomaly_results": {...}
}
```

### Export Results
```http
POST /api/export
Content-Type: application/json

{
  "forecast_data": [...]
}

Response: CSV file download
```

### Health Check
```http
GET /health

Response: {
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "modules_available": true
}
```

## 🧪 Testing

### Manual Testing Checklist

1. **File Upload**
   - ✅ Valid CSV files upload successfully
   - ✅ Invalid files show appropriate errors
   - ✅ File size limits enforced
   - ✅ Column validation works

2. **Forecast Generation**
   - ✅ Regression models generate forecasts
   - ✅ ARIMA models generate forecasts
   - ✅ Model comparison works correctly
   - ✅ Anomaly detection runs successfully

3. **Results Display**
   - ✅ Model performance metrics display
   - ✅ Charts render correctly
   - ✅ Tables populate with data
   - ✅ Export functionality works

4. **Error Handling**
   - ✅ Network errors handled gracefully
   - ✅ Invalid data shows user-friendly errors
   - ✅ Loading states work correctly
   - ✅ Toast notifications appear

### Automated Testing
```bash
# Run basic health check
curl -f http://localhost:5000/health

# Test file upload (requires test file)
curl -X POST -F "file=@test_data.csv" http://localhost:5000/api/upload
```

## 🐛 Troubleshooting

### Common Issues

**1. Models not loading**
```bash
# Check if model files exist
ls -la models/regression/
ls -la models/arima/

# Retrain models if missing
python inventory_forecasting_regression.py data/inventory_delivery_forecast_data.csv
```

**2. Memory issues during training**
```bash
# Monitor memory usage
docker stats

# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory
```

**3. File upload failures**
```bash
# Check upload directory permissions
chmod 755 uploads/

# Check file size limits in app.py
# MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
```

**4. Port conflicts**
```bash
# Check if port 5000 is in use
lsof -i :5000

# Use different port
docker run -p 8080:5000 restaurant-inventory-app
```

### Debug Mode

Enable debug mode for development:
```bash
# Set environment variable
export FLASK_ENV=development

# Or modify app.py
app.run(host='0.0.0.0', port=5000, debug=True)
```

### Logs and Monitoring

```bash
# View application logs
docker-compose logs -f restaurant-inventory-app

# View specific container logs
docker logs <container-id>

# Monitor system resources
docker stats
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Guidelines
- Follow PEP 8 for Python code
- Use meaningful commit messages
- Add tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn** for machine learning models
- **Flask** for web framework
- **Chart.js** for interactive visualizations
- **Docker** for containerization
- **Bootstrap** inspiration for responsive design

## 📞 Support

For support and questions:
- 📧 Email: [your-email@domain.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 Documentation: [Project Wiki](https://github.com/your-repo/wiki)

---

**Built with ❤️ for restaurant managers who want to make data-driven inventory decisions.**

## 🔄 Version History

- **v1.0.0** - Initial release with regression and ARIMA models
- **v1.1.0** - Added anomaly detection and improved UI
- **v1.2.0** - Docker containerization and cloud deployment support
- **v1.3.0** - Enhanced performance and production optimizations

---

*Last updated: January 2025*