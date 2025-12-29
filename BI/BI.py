import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class BusinessIntelligence:
    """Main class for Business Intelligence operations"""
    
    def __init__(self, data=None, data_path=None):
        """
        Initialize BI tool with data
        
        Args:
            data: pandas DataFrame (optional)
            data_path: path to CSV/Excel file (optional)
        """
        if data is not None:
            self.df = data.copy()
        elif data_path is not None:
            self.df = self.load_data(data_path)
        else:
            # Create sample data for demonstration
            self.df = self.generate_sample_data()
        
        self.clean_data()
        print(f"Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
    
    def load_data(self, file_path):
        """
        Load data from CSV or Excel file
        
        Args:
            file_path: path to data file
            
        Returns:
            pandas DataFrame
        """
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
    
    def generate_sample_data(self):
        """Generate sample sales data for demonstration"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        data = {
            'Date': np.random.choice(dates, size=1000),
            'Product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], size=1000),
            'Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], size=1000),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], size=1000),
            'Sales': np.random.uniform(100, 5000, size=1000),
            'Quantity': np.random.randint(1, 50, size=1000),
            'Customer_ID': np.random.randint(1000, 9999, size=1000)
        }
        
        df = pd.DataFrame(data)
        df['Revenue'] = df['Sales'] * df['Quantity']
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Year'] = df['Date'].dt.year
        
        return df
    
    def clean_data(self):
        """Clean and preprocess the data"""
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        
        # Remove outliers (optional - using IQR method)
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
    
    def descriptive_analysis(self):
        """Perform descriptive statistical analysis"""
        print("\n" + "="*60)
        print("DESCRIPTIVE ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print("\n1. Basic Statistics:")
        print(self.df.describe())
        
        # Data types and info
        print("\n2. Data Types:")
        print(self.df.dtypes)
        
        # Missing values
        print("\n3. Missing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values")
        
        return self.df.describe()
    
    def calculate_kpis(self):
        """Calculate Key Performance Indicators"""
        print("\n" + "="*60)
        print("KEY PERFORMANCE INDICATORS (KPIs)")
        print("="*60)
        
        kpis = {}
        
        # Revenue metrics
        if 'Revenue' in self.df.columns:
            kpis['Total Revenue'] = self.df['Revenue'].sum()
            kpis['Average Revenue'] = self.df['Revenue'].mean()
            kpis['Revenue Growth'] = self.calculate_growth_rate('Revenue')
        
        # Sales metrics
        if 'Sales' in self.df.columns:
            kpis['Total Sales'] = self.df['Sales'].sum()
            kpis['Average Sales'] = self.df['Sales'].mean()
            kpis['Max Sales'] = self.df['Sales'].max()
            kpis['Min Sales'] = self.df['Sales'].min()
        
        # Quantity metrics
        if 'Quantity' in self.df.columns:
            kpis['Total Quantity'] = self.df['Quantity'].sum()
            kpis['Average Quantity'] = self.df['Quantity'].mean()
        
        # Display KPIs
        for key, value in kpis.items():
            if isinstance(value, float):
                print(f"{key}: ${value:,.2f}" if 'Revenue' in key or 'Sales' in key else f"{key}: {value:,.2f}")
            else:
                print(f"{key}: {value}")
        
        return kpis
    
    def calculate_growth_rate(self, column):
        """Calculate growth rate for a numeric column"""
        if 'Date' not in self.df.columns:
            return None
        
        # Group by month and calculate growth
        monthly = self.df.groupby(self.df['Date'].dt.to_period('M'))[column].sum()
        if len(monthly) > 1:
            growth = ((monthly.iloc[-1] - monthly.iloc[0]) / monthly.iloc[0]) * 100
            return growth
        return None
    
    def analyze_by_category(self, category_col='Category', value_col='Revenue'):
        """Analyze data by category"""
        print("\n" + "="*60)
        print(f"ANALYSIS BY {category_col.upper()}")
        print("="*60)
        
        category_analysis = self.df.groupby(category_col)[value_col].agg([
            'sum', 'mean', 'count', 'min', 'max'
        ]).round(2)
        
        category_analysis.columns = ['Total', 'Average', 'Count', 'Min', 'Max']
        category_analysis = category_analysis.sort_values('Total', ascending=False)
        
        print(category_analysis)
        return category_analysis
    
    def analyze_by_region(self, region_col='Region', value_col='Revenue'):
        """Analyze data by region"""
        print("\n" + "="*60)
        print(f"ANALYSIS BY {region_col.upper()}")
        print("="*60)
        
        region_analysis = self.df.groupby(region_col)[value_col].agg([
            'sum', 'mean', 'count'
        ]).round(2)
        
        region_analysis.columns = ['Total', 'Average', 'Count']
        region_analysis = region_analysis.sort_values('Total', ascending=False)
        
        print(region_analysis)
        return region_analysis
    
    def time_series_analysis(self, date_col='Date', value_col='Revenue'):
        """Perform time series analysis"""
        print("\n" + "="*60)
        print("TIME SERIES ANALYSIS")
        print("="*60)
        
        if date_col not in self.df.columns:
            print("Date column not found!")
            return None
        
        # Set date as index
        df_time = self.df.set_index(date_col)
        
        # Daily aggregation
        daily = df_time[value_col].resample('D').sum()
        print(f"\nDaily {value_col}:")
        print(f"Average: ${daily.mean():,.2f}")
        print(f"Total: ${daily.sum():,.2f}")
        
        # Monthly aggregation
        monthly = df_time[value_col].resample('M').sum()
        print(f"\nMonthly {value_col}:")
        print(monthly)
        
        # Quarterly aggregation
        quarterly = df_time[value_col].resample('Q').sum()
        print(f"\nQuarterly {value_col}:")
        print(quarterly)
        
        return {
            'daily': daily,
            'monthly': monthly,
            'quarterly': quarterly
        }
    
    def visualize_sales_trends(self, date_col='Date', value_col='Revenue'):
        """Create line chart for sales trends"""
        if date_col not in self.df.columns:
            print("Date column not found for trend analysis!")
            return
        
        df_time = self.df.set_index(date_col)
        monthly = df_time[value_col].resample('M').sum()
        
        plt.figure(figsize=(14, 6))
        plt.plot(monthly.index, monthly.values, marker='o', linewidth=2, markersize=8)
        plt.title(f'{value_col} Trends Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(value_col, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def visualize_category_performance(self, category_col='Category', value_col='Revenue'):
        """Create bar chart for category performance"""
        category_data = self.df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(category_data.index, category_data.values, color='steelblue', alpha=0.7)
        plt.title(f'{value_col} by {category_col}', fontsize=16, fontweight='bold')
        plt.xlabel(category_col, fontsize=12)
        plt.ylabel(value_col, fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_region_distribution(self, region_col='Region', value_col='Revenue'):
        """Create pie chart for region distribution"""
        region_data = self.df.groupby(region_col)[value_col].sum()
        
        plt.figure(figsize=(10, 8))
        colors = sns.color_palette("pastel")
        plt.pie(region_data.values, labels=region_data.index, autopct='%1.1f%%',
                startangle=90, colors=colors)
        plt.title(f'{value_col} Distribution by {region_col}', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def visualize_correlation_heatmap(self):
        """Create correlation heatmap for numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            print("Not enough numeric columns for correlation analysis!")
            return
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_dashboard(self):
        """Create a comprehensive dashboard with multiple visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Business Intelligence Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Sales trends
        if 'Date' in self.df.columns:
            df_time = self.df.set_index('Date')
            monthly = df_time['Revenue'].resample('M').sum()
            axes[0, 0].plot(monthly.index, monthly.values, marker='o', linewidth=2)
            axes[0, 0].set_title('Monthly Revenue Trends', fontweight='bold')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Revenue')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Category performance
        if 'Category' in self.df.columns:
            category_data = self.df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)
            axes[0, 1].bar(category_data.index, category_data.values, color='steelblue', alpha=0.7)
            axes[0, 1].set_title('Revenue by Category', fontweight='bold')
            axes[0, 1].set_xlabel('Category')
            axes[0, 1].set_ylabel('Revenue')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Region distribution
        if 'Region' in self.df.columns:
            region_data = self.df.groupby('Region')['Revenue'].sum()
            axes[1, 0].pie(region_data.values, labels=region_data.index, autopct='%1.1f%%',
                           startangle=90)
            axes[1, 0].set_title('Revenue Distribution by Region', fontweight='bold')
        
        # 4. Top products
        if 'Product' in self.df.columns:
            product_data = self.df.groupby('Product')['Revenue'].sum().nlargest(5)
            axes[1, 1].barh(product_data.index, product_data.values, color='coral', alpha=0.7)
            axes[1, 1].set_title('Top 5 Products by Revenue', fontweight='bold')
            axes[1, 1].set_xlabel('Revenue')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, output_path='BI_report.txt'):
        """Generate a text report with all analyses"""
        report = []
        report.append("="*60)
        report.append("BUSINESS INTELLIGENCE REPORT")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*60)
        
        # Basic info
        report.append(f"\nDataset Information:")
        report.append(f"Total Records: {len(self.df)}")
        report.append(f"Total Columns: {len(self.df.columns)}")
        report.append(f"Date Range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        # KPIs
        report.append("\n" + "="*60)
        report.append("KEY PERFORMANCE INDICATORS")
        report.append("="*60)
        kpis = self.calculate_kpis()
        for key, value in kpis.items():
            if isinstance(value, float):
                report.append(f"{key}: ${value:,.2f}" if 'Revenue' in key or 'Sales' in key else f"{key}: {value:,.2f}")
            else:
                report.append(f"{key}: {value}")
        
        # Category analysis
        if 'Category' in self.df.columns:
            report.append("\n" + "="*60)
            report.append("CATEGORY ANALYSIS")
            report.append("="*60)
            cat_analysis = self.analyze_by_category()
            report.append(str(cat_analysis))
        
        # Region analysis
        if 'Region' in self.df.columns:
            report.append("\n" + "="*60)
            report.append("REGION ANALYSIS")
            report.append("="*60)
            region_analysis = self.analyze_by_region()
            report.append(str(region_analysis))
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\nReport saved to: {output_path}")
        return report


def main():
    """Main function to run BI analysis"""
    print("="*60)
    print("BUSINESS INTELLIGENCE - DATA ANALYSIS TOOL")
    print("="*60)
    
    # Initialize BI tool (will use sample data if no file provided)
    # Option 1: Use sample data
    bi = BusinessIntelligence()
    
    # Option 2: Load from file (uncomment and provide path)
    # bi = BusinessIntelligence(data_path='data/sales_data.csv')
    
    # Perform analyses
    bi.descriptive_analysis()
    bi.calculate_kpis()
    
    if 'Category' in bi.df.columns:
        bi.analyze_by_category()
    
    if 'Region' in bi.df.columns:
        bi.analyze_by_region()
    
    if 'Date' in bi.df.columns:
        bi.time_series_analysis()
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS...")
    print("="*60)
    
    if 'Date' in bi.df.columns:
        bi.visualize_sales_trends()
    
    if 'Category' in bi.df.columns:
        bi.visualize_category_performance()
    
    if 'Region' in bi.df.columns:
        bi.visualize_region_distribution()
    
    bi.visualize_correlation_heatmap()
    
    # Create comprehensive dashboard
    print("\nGenerating Dashboard...")
    bi.create_dashboard()
    
    # Generate report
    print("\n" + "="*60)
    print("GENERATING REPORT...")
    print("="*60)
    bi.generate_report()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

