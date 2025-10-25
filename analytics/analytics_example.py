"""
Example usage of the Analytics feature
Demonstrates various ways to use charts and insights
"""

import pandas as pd
import asyncio
from analytics.analytics_handler import AnalyticsHandler


async def example_sales_analysis():
    """Example: Analyze sales data"""
    
    # Sample sales data
    data = {
        "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "revenue": [50000, 55000, 60000, 58000, 65000, 70000],
        "expenses": [30000, 32000, 35000, 33000, 36000, 38000],
        "customers": [450, 480, 520, 500, 550, 580]
    }
    
    handler = AnalyticsHandler()
    
    # Generate both chart and insights
    results = await handler.process_analytics_request(
        request_type="both",
        data=data,
        params={
            "chart_type": "line",
            "title": "Sales Performance - H1 2024",
            "x": "month",
            "y": "revenue",
            "context": "First half sales data for 2024",
            "focus_areas": ["trends", "growth"]
        }
    )
    
    print("Sales Analysis Results:")
    print(f"Chart generated: {results.get('chart', {}).get('success')}")
    print(f"Insights generated: {results.get('insights', {}).get('success')}")
    
    if results.get('insights', {}).get('success'):
        insights = results['insights']['data']
        print(f"\nKey Findings: {insights.get('key_findings')}")
        print(f"\nInsights: {insights.get('insights')}")
        print(f"\nRecommendations: {insights.get('recommendations')}")


async def example_multiple_series():
    """Example: Create multi-series chart"""
    
    handler = AnalyticsHandler()
    chart_gen = handler.chart_gen
    
    data = {
        "x": ["Q1", "Q2", "Q3", "Q4"],
        "Product_A": [100, 120, 140, 160],
        "Product_B": [80, 95, 110, 125],
        "Product_C": [60, 75, 85, 100]
    }
    
    fig = chart_gen.create_multi_series_chart(
        chart_type="bar",
        data=data,
        series_names=["Product_A", "Product_B", "Product_C"],
        title="Quarterly Sales by Product"
    )
    
    print("Multi-series chart created successfully")
    return fig


async def example_statistical_insights():
    """Example: Get detailed statistical insights"""
    
    # Generate random data for demonstration
    import numpy as np
    
    data = pd.DataFrame({
        "metric": np.random.randn(100) * 10 + 50,
        "category": np.random.choice(["A", "B", "C"], 100),
        "value": np.random.randint(1, 100, 100)
    })
    
    handler = AnalyticsHandler()
    
    results = await handler.process_analytics_request(
        request_type="insights",
        data=data,
        params={
            "context": "Performance metrics across categories",
            "focus_areas": ["outliers", "correlations", "distribution"]
        }
    )
    
    if results.get('insights', {}).get('success'):
        insights_data = results['insights']['data']
        
        print("\n=== Statistical Insights ===")
        print("\nStatistics:")
        for col, stats in insights_data.get('statistics', {}).items():
            print(f"\n{col}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Median: {stats['median']:.2f}")
            print(f"  Std Dev: {stats['std']:.2f}")
            print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        
        print("\nPatterns Detected:")
        patterns = insights_data.get('patterns', {})
        
        if patterns.get('trends'):
            print("\nTrends:")
            for trend in patterns['trends']:
                print(f"  - {trend['column']}: {trend['direction']} (strength: {trend['strength']:.2f})")
        
        if patterns.get('outliers'):
            print("\nOutliers:")
            for outlier in patterns['outliers']:
                print(f"  - {outlier['column']}: {outlier['count']} outliers detected")
        
        if patterns.get('correlations'):
            print("\nCorrelations:")
            for corr in patterns['correlations']:
                print(f"  - {corr['columns'][0]} & {corr['columns'][1]}: {corr['strength']} ({corr['correlation']:.2f})")


async def example_custom_chart_types():
    """Example: Create different chart types"""
    
    handler = AnalyticsHandler()
    chart_gen = handler.chart_gen
    
    # Pie chart
    pie_data = {
        "category": ["Marketing", "Operations", "R&D", "Sales"],
        "budget": [30000, 45000, 25000, 50000]
    }
    
    pie_fig = chart_gen.create_chart(
        chart_type="pie",
        data=pie_data,
        title="Budget Allocation",
        values="budget",
        names="category"
    )
    
    print("Pie chart created")
    
    # Histogram
    hist_data = pd.DataFrame({
        "age": [25, 30, 35, 28, 42, 38, 45, 32, 29, 50, 27, 33, 41, 36, 31]
    })
    
    hist_fig = chart_gen.create_chart(
        chart_type="histogram",
        data=hist_data,
        title="Age Distribution",
        x="age"
    )
    
    print("Histogram created")
    
    # Box plot
    box_data = pd.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        "values": list(range(10)) + list(range(15, 25)) + list(range(5, 15))
    })
    
    box_fig = chart_gen.create_chart(
        chart_type="box",
        data=box_data,
        title="Value Distribution by Group",
        x="group",
        y="values"
    )
    
    print("Box plot created")
    
    return pie_fig, hist_fig, box_fig


async def example_csv_file_analysis():
    """Example: Analyze CSV file"""
    
    # Create sample CSV
    data = {
        "date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "visitors": [100 + i * 5 + (i % 7) * 10 for i in range(30)],
        "conversions": [10 + i // 3 + (i % 5) * 2 for i in range(30)],
        "revenue": [1000 + i * 50 + (i % 7) * 100 for i in range(30)]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("sample_analytics.csv", index=False)
    
    # Analyze the CSV
    handler = AnalyticsHandler()
    
    results = await handler.process_analytics_request(
        request_type="both",
        data="sample_analytics.csv",
        params={
            "chart_type": "area",
            "title": "Website Analytics - January 2024",
            "x": "date",
            "y": "visitors",
            "context": "Daily website traffic and conversion data"
        }
    )
    
    print("\nCSV Analysis Complete")
    print(f"Insights: {results.get('insights', {}).get('data', {}).get('key_findings')}")


async def main():
    """Run all examples"""
    print("=== Analytics Feature Examples ===\n")
    
    print("\n1. Sales Analysis")
    await example_sales_analysis()
    
    print("\n2. Multiple Series Chart")
    await example_multiple_series()
    
    print("\n3. Statistical Insights")
    await example_statistical_insights()
    
    print("\n4. Custom Chart Types")
    await example_custom_chart_types()
    
    print("\n5. CSV File Analysis")
    await example_csv_file_analysis()
    
    print("\n=== All Examples Complete ===")


if __name__ == "__main__":
    asyncio.run(main())