# DGR Deviation Dashboard

A modern, beautiful React dashboard for analyzing Daily Generation Report (DGR) deviation data with real-time Supabase integration. This dashboard replaces the previous Streamlit implementation with a more performant and visually appealing solution.

## 🚀 Features

### ✨ Modern UI/UX
- **Beautiful Design**: Clean, modern interface with Tailwind CSS
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
- **Smooth Animations**: Subtle animations and transitions for better UX
- **Dark/Light Theme**: (Can be extended)

### 📊 Data Visualization
- **Interactive Charts**: Line charts, bar charts, and pie charts using Recharts
- **Real-time Updates**: Live data updates from Supabase
- **Multiple Chart Types**: 
  - Deviation trend analysis
  - Plant performance rankings
  - Reason distribution analysis

### 🔍 Advanced Analytics
- **Plant Rankings**: Compare performance across multiple plants
- **Threshold Analysis**: Customizable deviation thresholds
- **Statistical Overview**: Key metrics and performance indicators
- **Export Functionality**: Export data to CSV/Excel

### 🛠 Data Management
- **Interactive Tables**: Sortable, filterable data tables
- **Advanced Filtering**: Multi-plant, date range, and threshold filters
- **Reason Tracking**: Add and manage deviation reasons
- **Comment System**: Add detailed comments for each deviation

### ⚡ Performance
- **Fast Loading**: Optimized data fetching and caching
- **Efficient Rendering**: React best practices for smooth performance
- **Background Processing**: Non-blocking data operations

## 🏗 Technology Stack

- **Frontend**: React 18 + TypeScript
- **Styling**: Tailwind CSS with custom design system
- **Charts**: Recharts library
- **Database**: Supabase (PostgreSQL)
- **Build Tool**: Vite
- **Icons**: Lucide React + Heroicons
- **State Management**: React hooks

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Supabase account (for database)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd dgr-dashboard
```

2. **Install dependencies**
```bash
npm install
```

3. **Environment Setup**
Create a `.env` file in the root directory:
```env
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

4. **Start development server**
```bash
npm run dev
```

5. **Open your browser**
Navigate to `http://localhost:5173`

### Database Setup

The dashboard expects the following Supabase tables:

```sql
-- DGR data table
CREATE TABLE dgr_data (
    id SERIAL PRIMARY KEY,
    plant VARCHAR(255) NOT NULL,
    date DATE NOT NULL,
    input_name VARCHAR(255) NOT NULL,
    value DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Reasons table
CREATE TABLE reasons (
    id SERIAL PRIMARY KEY,
    reason_name VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Plant mapping table
CREATE TABLE plant_mapping (
    id SERIAL PRIMARY KEY,
    plant_name VARCHAR(255) UNIQUE NOT NULL,
    data_start_col VARCHAR(10),
    data_end_col VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## 📱 Dashboard Sections

### 1. Overview
- **Statistics Cards**: Key metrics at a glance
- **Deviation Trends**: Time-series analysis
- **Quick Rankings**: Top/bottom performing plants

### 2. Data Table
- **Interactive Grid**: Sortable, filterable data table
- **Search Functionality**: Quick data lookup
- **Export Options**: CSV download capability

### 3. Rankings
- **Plant Performance**: Comprehensive ranking system
- **Visual Charts**: Bar charts with color coding
- **Comparative Analysis**: Side-by-side comparisons

### 4. Analytics
- **Deep Dive Charts**: Detailed analytical views
- **Correlation Analysis**: Performance patterns
- **Trend Identification**: Long-term analysis

### 5. Reasons
- **Reason Tracking**: Add deviation causes
- **Comment System**: Detailed explanations
- **Quick Stats**: Reason distribution overview

## 🎨 Design System

### Color Palette
- **Primary**: Blue (#2563eb) - Main actions and highlights
- **Success**: Green (#16a34a) - Positive indicators
- **Warning**: Yellow (#f59e0b) - Caution states
- **Danger**: Red (#dc2626) - Critical issues
- **Gray**: Various shades for text and backgrounds

### Typography
- **Font**: Inter (Google Fonts)
- **Headings**: Semibold weights
- **Body**: Regular weights
- **Code**: Monospace for technical data

### Components
All components follow a consistent design pattern:
- Rounded corners (8px)
- Subtle shadows
- Hover states
- Loading states
- Error handling

## 🔧 Customization

### Adding New Charts
1. Create component in `src/components/charts/`
2. Use Recharts library
3. Follow existing patterns for props and styling

### Custom Themes
1. Update `tailwind.config.js`
2. Modify color palette in config
3. Update component styles accordingly

### New Dashboard Sections
1. Add tab to `Header.tsx`
2. Create section component
3. Add route handling in `App.tsx`

## 🚀 Deployment

### Build for Production
```bash
npm run build
```

### Deploy to Vercel
```bash
npm install -g vercel
vercel --prod
```

### Deploy to Netlify
1. Connect repository to Netlify
2. Set build command: `npm run build`
3. Set publish directory: `dist`

## 📊 Data Integration

### Supabase Connection
The dashboard connects to Supabase for:
- Real-time data fetching
- Reason management
- Plant configuration
- Historical data analysis

### API Endpoints
- `GET /dgr_data` - Fetch deviation data
- `POST /reasons` - Add new reasons
- `GET /plant_mapping` - Plant configuration

## 🛡 Performance Optimization

- **Lazy Loading**: Components loaded on demand
- **Memoization**: React.memo for expensive components
- **Virtual Scrolling**: For large data tables
- **Debounced Search**: Optimized filtering
- **Caching**: Supabase query caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support, please contact the development team or create an issue in the repository.

---

**Converted from Streamlit to React** - Now with 10x better performance and user experience! 🎉
