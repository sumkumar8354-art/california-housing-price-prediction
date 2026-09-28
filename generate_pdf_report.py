import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (Top)
        self.drawString(36, letter[1] - 24, "MainCraft Technology — AI & Machine Learning Task 2")
        self.drawRightString(letter[0] - 36, letter[1] - 24, "Feature Engineering & Model Optimization")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.6)
        self.line(36, letter[1] - 28, letter[0] - 36, letter[1] - 28)
        
        # Footer (Bottom)
        self.line(36, 30, letter[0] - 36, 30)
        self.drawString(36, 20, "Confidential — MainCraft Technology AI/ML Intern Task Evaluation")
        self.drawRightString(letter[0] - 36, 20, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf():
    pdf_filename = "AI_ML_Task2_Report.pdf"
    
    # 36 pt (0.5 in) margins
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#0f2942"),
        spaceAfter=2
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1e40af"),
        spaceAfter=6
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#4a5568")
    )
    
    section_h1 = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=14.5,
        textColor=colors.HexColor("#0f2942"),
        spaceBefore=6,
        spaceAfter=3
    )
    
    section_h2 = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=3,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.3,
        leading=11.2,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        alignment=1
    )
    
    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        alignment=0
    )

    story = []

    # ==================== PAGE 1 ====================
    # Title & Subtitle
    story.append(Paragraph("Artificial Intelligence & Machine Learning — Task 2", title_style))
    story.append(Paragraph("Feature Engineering, Model Optimization & Performance Comparison Report", subtitle_style))
    
    # Metadata bar
    meta_text = "<b>Organization:</b> MainCraft Technology &nbsp;|&nbsp; <b>Project:</b> Enhanced House Price Prediction &nbsp;|&nbsp; <b>Dataset:</b> California Housing (Census)"
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f2942"), spaceBefore=2, spaceAfter=5))
    
    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", section_h1))
    exec_summary_text = (
        "In enterprise machine learning workflows, raw algorithms rarely perform adequately without systematic "
        "feature engineering, normalization, and comparative evaluation. This study develops an end-to-end House "
        "Price Prediction system using the benchmark <b>California Housing dataset</b> (20,640 records). "
        "We implemented and evaluated three distinct learning architectures: <b>Ordinary Least Squares (OLS) Linear Regression</b> (baseline), "
        "<b>Ridge Regression</b> (L2 regularization), and <b>Decision Tree Regressor</b> (non-linear tree structure). "
        "Standardization via <code>StandardScaler</code> was enforced across all 8 continuous predictors. "
        "The empirical findings demonstrate that the non-linear Decision Tree model achieved the best overall performance, "
        "attaining a <b>Root Mean Squared Error (RMSE) of 0.7242</b> and an <b>R² score of 0.5997</b>, outperforming linear "
        "baselines by capturing non-linear interactions across socioeconomic and spatial attributes."
    )
    story.append(Paragraph(exec_summary_text, body_style))
    
    # 2. Methodology & System Architecture
    story.append(Paragraph("2. Methodology & System Architecture", section_h1))
    
    story.append(Paragraph("<b>2.1 Dataset Ingestion & Preprocessing</b>", section_h2))
    data_prep_text = (
        "The dataset comprises 20,640 block groups with 8 continuous predictors: median income (<code>MedInc</code>), housing age (<code>HouseAge</code>), "
        "average rooms (<code>AveRooms</code>), bedrooms (<code>AveBedrms</code>), population (<code>Population</code>), average occupancy (<code>AveOccup</code>), "
        "and geographical coordinates (<code>Latitude</code>, <code>Longitude</code>). The response target is median house value (<code>HousePrice</code>, in $100k units). "
        "Initial exploratory analysis verified complete data integrity with zero missing values."
    )
    story.append(Paragraph(data_prep_text, body_style))
    
    story.append(Paragraph("<b>2.2 Feature Scaling (StandardScaler) — Critical Step</b>", section_h2))
    scaling_text = (
        "Raw features exhibit stark scale disparities (e.g., <code>Population</code> spans up to 35,682, whereas <code>MedInc</code> ranges from 0.49 to 15.00). "
        "Failing to standardize features results in gradient instability, numerical dominance, and biased penalties in regularized models. "
        "All features were normalized using z-score standardization: <i>z = (x - μ) / σ</i>, centering distributions to mean zero (μ ≈ 0) and standard deviation one (σ = 1)."
    )
    story.append(Paragraph(scaling_text, body_style))
    
    story.append(Paragraph("<b>2.3 Partitioning & Algorithm Selection</b>", section_h2))
    partition_text = (
        "An 80/20 train/test split (16,512 train samples, 4,128 test samples) with <code>random_state=42</code> was instituted for unbiased evaluation. "
        "Three complementary models were trained: (1) <b>Linear Regression:</b> Parametric baseline assuming additive linearity; "
        "(2) <b>Ridge Regression (α=1.0):</b> Penalized objective function <i>min ||Xw - y||² + α||w||²</i> to mitigate multicollinearity; and "
        "(3) <b>Decision Tree Regressor (max_depth=5):</b> Non-parametric recursive splitting model capturing non-linear thresholds and interactions."
    )
    story.append(Paragraph(partition_text, body_style))
    
    # 3. Model Performance Comparison
    story.append(Paragraph("3. Quantitative Performance Comparison", section_h1))
    
    # Comparison Table
    table_data = [
        [
            Paragraph("<b>Model Algorithm</b>", table_header_style),
            Paragraph("<b>RMSE ($100k)<br/>(Lower = Better)</b>", table_header_style),
            Paragraph("<b>R² Score<br/>(Higher = Better)</b>", table_header_style),
            Paragraph("<b>Relative Performance & Architectural Notes</b>", table_header_style)
        ],
        [
            Paragraph("<b>Linear Regression</b>", table_cell_left),
            Paragraph("0.7456", table_cell_style),
            Paragraph("0.5758", table_cell_style),
            Paragraph("Baseline linear model; constrained by strict additive linearity assumption.", table_cell_left)
        ],
        [
            Paragraph("<b>Ridge Regression (α=1.0)</b>", table_cell_left),
            Paragraph("0.7456", table_cell_style),
            Paragraph("0.5758", table_cell_style),
            Paragraph("L2 regularized linear model; near-identical metrics due to high N / P ratio.", table_cell_left)
        ],
        [
            Paragraph("<b>Decision Tree (max_depth=5)</b>", table_cell_left),
            Paragraph("<b>0.7242</b>", table_cell_style),
            Paragraph("<b>0.5997</b>", table_cell_style),
            Paragraph("<b>Top Performer:</b> Captures non-linear thresholds & geographic coordinates.", table_cell_left)
        ]
    ]
    
    comp_table = Table(table_data, colWidths=[1.4*inch, 1.2*inch, 1.1*inch, 3.5*inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f2942")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, 2), [colors.HexColor("#F8FAFC"), colors.HexColor("#FFFFFF")]),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#E6FFFA")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 5))

    # Metric Comparison Chart on Page 1
    story.append(Image("figures/model_comparison.png", width=6.8*inch, height=2.35*inch))
    
    # ==================== PAGE 2 ====================
    story.append(PageBreak())
    
    # 4. Visual Diagnostics
    story.append(Paragraph("4. Visual Diagnostics & In-Depth Validation", section_h1))
    
    story.append(Paragraph("<b>4.1 Actual vs. Predicted Validation</b>", section_h2))
    story.append(Image("figures/actual_vs_predicted.png", width=6.8*inch, height=2.4*inch))
    story.append(Spacer(1, 3))
    
    diag_text = (
        "Scatter plots comparing predictions against true values (above) highlight key operational differences. "
        "The red dashed line represents perfect calibration (<i>y = x</i>). While Linear Regression exhibits wider variance "
        "and projects negative values for lower-bracket properties, the Decision Tree bounds predictions more realistically. "
        "Furthermore, residual error distribution analysis confirms zero-centered errors with reduced tail dispersion."
    )
    story.append(Paragraph(diag_text, body_style))
    story.append(Spacer(1, 3))
    
    # Feature Importance Section
    story.append(Paragraph("<b>4.2 Decision Tree Feature Importance & Economic Drivers</b>", section_h2))
    story.append(Image("figures/feature_importance.png", width=4.6*inch, height=2.1*inch))
    story.append(Spacer(1, 2))
    
    fi_text = (
        "Feature importance scoring measures the variance reduction contribution per feature across tree splits. "
        "<b>Median Income (<code>MedInc</code>) is the dominant predictor, accounting for 77.1% of total split importance</b>. "
        "Average occupancy (<code>AveOccup</code>, 12.8%), housing age (<code>HouseAge</code>, 4.2%), and rooms (<code>AveRooms</code>, 3.1%) "
        "form the secondary factors, corroborating real-world real estate economics."
    )
    story.append(Paragraph(fi_text, body_style))
    
    # 5. Technical Discussion & Conclusion
    story.append(Paragraph("5. Technical Discussion & Conclusion", section_h1))
    discussion_text = (
        "• <b>Linearity vs. Non-Linearity:</b> Linear Regression and Ridge Regression attained identical scores (R² = 0.5758, RMSE = 0.7456). "
        "Because sample size (N = 16,512) substantially exceeds feature count (P = 8), high variance did not arise, rendering L2 shrinkage minimal. "
        "However, housing prices involve non-linear geographic and demographic boundaries that the Decision Tree successfully segmented, raising R² to 0.5997.<br/>"
        "• <b>Production Serialization:</b> The top-performing Decision Tree model and fitted <code>StandardScaler</code> "
        "were serialized as <code>best_model.joblib</code> and <code>scaler.joblib</code>. Production inference pipelines were verified on test vectors, "
        "confirming sub-millisecond latency and deployment readiness.<br/>"
        "• <b>Future Enhancements:</b> Recommendations include ensembling (Random Forests, XGBoost) and non-linear spatial feature engineering."
    )
    story.append(Paragraph(discussion_text, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report successfully compiled to {pdf_filename}")

if __name__ == "__main__":
    build_pdf()
