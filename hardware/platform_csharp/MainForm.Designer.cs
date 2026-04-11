namespace WinFormsApp_Draft
{
    partial class MainForm
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            components = new System.ComponentModel.Container();
            Response = new TextBox();
            SpinTimer = new System.Windows.Forms.Timer(components);
            AutoRead = new CheckBox();
            ControlPanel = new Panel();
            OpenArmForm = new Button();
            OpenCoaterForm = new Button();
            OpenDispenserForm = new Button();
            MoveTest = new Button();
            Refresh = new Button();
            FilePath = new TextBox();
            SheetName = new TextBox();
            label1 = new Label();
            label2 = new Label();
            ReagentLayout = new TableLayoutPanel();
            ExperimentParameters = new ListBox();
            ReagentFeatures = new ListBox();
            Method = new GroupBox();
            label6 = new Label();
            NameReagent = new TextBox();
            Log = new RichTextBox();
            label4 = new Label();
            label3 = new Label();
            Tipbox_Reload = new Button();
            Substrate_Reload = new Button();
            OpenSpecForm = new Button();
            AI_Agent = new Button();
            ClearConf = new Label();
            ClearReagent = new Button();
            label5 = new Label();
            Method.SuspendLayout();
            SuspendLayout();
            // 
            // Response
            // 
            Response.Location = new Point(134, 22);
            Response.Margin = new Padding(2, 3, 2, 3);
            Response.Name = "Response";
            Response.ReadOnly = true;
            Response.Size = new Size(421, 23);
            Response.TabIndex = 17;
            // 
            // AutoRead
            // 
            AutoRead.AutoSize = true;
            AutoRead.Location = new Point(237, 135);
            AutoRead.Name = "AutoRead";
            AutoRead.Size = new Size(84, 21);
            AutoRead.TabIndex = 47;
            AutoRead.Text = "auto read";
            AutoRead.UseVisualStyleBackColor = true;
            AutoRead.CheckedChanged += AutoRead_CheckedChanged;
            // 
            // ControlPanel
            // 
            ControlPanel.BackColor = SystemColors.AppWorkspace;
            ControlPanel.Location = new Point(581, 57);
            ControlPanel.Name = "ControlPanel";
            ControlPanel.Size = new Size(575, 689);
            ControlPanel.TabIndex = 50;
            // 
            // OpenArmForm
            // 
            OpenArmForm.Location = new Point(581, 12);
            OpenArmForm.Name = "OpenArmForm";
            OpenArmForm.Size = new Size(75, 23);
            OpenArmForm.TabIndex = 51;
            OpenArmForm.Text = "arm";
            OpenArmForm.UseVisualStyleBackColor = true;
            OpenArmForm.Click += OpenArmForm_Click;
            // 
            // OpenCoaterForm
            // 
            OpenCoaterForm.Location = new Point(662, 12);
            OpenCoaterForm.Name = "OpenCoaterForm";
            OpenCoaterForm.Size = new Size(75, 23);
            OpenCoaterForm.TabIndex = 52;
            OpenCoaterForm.Text = "coater";
            OpenCoaterForm.UseVisualStyleBackColor = true;
            OpenCoaterForm.Click += OpenCoaterForm_Click;
            // 
            // OpenDispenserForm
            // 
            OpenDispenserForm.Location = new Point(743, 12);
            OpenDispenserForm.Name = "OpenDispenserForm";
            OpenDispenserForm.Size = new Size(75, 23);
            OpenDispenserForm.TabIndex = 53;
            OpenDispenserForm.Text = "dispenser";
            OpenDispenserForm.UseVisualStyleBackColor = true;
            OpenDispenserForm.Click += OpenDispenserForm_Click;
            // 
            // MoveTest
            // 
            MoveTest.Location = new Point(264, 57);
            MoveTest.Name = "MoveTest";
            MoveTest.Size = new Size(94, 23);
            MoveTest.TabIndex = 54;
            MoveTest.Text = "Start Test";
            MoveTest.UseVisualStyleBackColor = true;
            MoveTest.Click += MoveTest_Click;
            // 
            // Refresh
            // 
            Refresh.Location = new Point(12, 12);
            Refresh.Name = "Refresh";
            Refresh.Size = new Size(84, 42);
            Refresh.TabIndex = 55;
            Refresh.Text = "Refresh Form";
            Refresh.UseVisualStyleBackColor = true;
            Refresh.Click += Refresh_Click;
            // 
            // FilePath
            // 
            FilePath.Location = new Point(195, 180);
            FilePath.Name = "FilePath";
            FilePath.Size = new Size(174, 23);
            FilePath.TabIndex = 48;
            // 
            // SheetName
            // 
            SheetName.Location = new Point(195, 220);
            SheetName.Name = "SheetName";
            SheetName.Size = new Size(174, 23);
            SheetName.TabIndex = 50;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(104, 183);
            label1.Name = "label1";
            label1.Size = new Size(58, 17);
            label1.TabIndex = 56;
            label1.Text = "file path:";
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.Location = new Point(84, 223);
            label2.Name = "label2";
            label2.Size = new Size(78, 17);
            label2.TabIndex = 57;
            label2.Text = "sheet name:";
            // 
            // ReagentLayout
            // 
            ReagentLayout.ColumnCount = 2;
            ReagentLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            ReagentLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            ReagentLayout.Location = new Point(40, 41);
            ReagentLayout.Name = "ReagentLayout";
            ReagentLayout.RowCount = 2;
            ReagentLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 50F));
            ReagentLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 50F));
            ReagentLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 20F));
            ReagentLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 20F));
            ReagentLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 20F));
            ReagentLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 20F));
            ReagentLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 20F));
            ReagentLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 20F));
            ReagentLayout.Size = new Size(208, 261);
            ReagentLayout.TabIndex = 58;
            // 
            // ExperimentParameters
            // 
            ExperimentParameters.FormattingEnabled = true;
            ExperimentParameters.ItemHeight = 17;
            ExperimentParameters.Location = new Point(18, 322);
            ExperimentParameters.Name = "ExperimentParameters";
            ExperimentParameters.Size = new Size(241, 106);
            ExperimentParameters.TabIndex = 59;
            // 
            // ReagentFeatures
            // 
            ReagentFeatures.FormattingEnabled = true;
            ReagentFeatures.ItemHeight = 17;
            ReagentFeatures.Location = new Point(280, 39);
            ReagentFeatures.Name = "ReagentFeatures";
            ReagentFeatures.Size = new Size(241, 89);
            ReagentFeatures.TabIndex = 60;
            // 
            // Method
            // 
            Method.Controls.Add(label5);
            Method.Controls.Add(ClearConf);
            Method.Controls.Add(ClearReagent);
            Method.Controls.Add(label6);
            Method.Controls.Add(NameReagent);
            Method.Controls.Add(Log);
            Method.Controls.Add(label4);
            Method.Controls.Add(label3);
            Method.Controls.Add(ReagentFeatures);
            Method.Controls.Add(ExperimentParameters);
            Method.Location = new Point(22, 275);
            Method.Name = "Method";
            Method.Size = new Size(533, 446);
            Method.TabIndex = 61;
            Method.TabStop = false;
            Method.Text = "Method";
            // 
            // label6
            // 
            label6.AutoSize = true;
            label6.Location = new Point(327, 131);
            label6.Name = "label6";
            label6.Size = new Size(147, 17);
            label6.TabIndex = 64;
            label6.Text = "Register Reagent Name";
            // 
            // NameReagent
            // 
            NameReagent.Location = new Point(280, 151);
            NameReagent.Name = "NameReagent";
            NameReagent.Size = new Size(241, 23);
            NameReagent.TabIndex = 63;
            // 
            // Log
            // 
            Log.Location = new Point(280, 232);
            Log.Name = "Log";
            Log.Size = new Size(241, 196);
            Log.TabIndex = 62;
            Log.Text = "";
            // 
            // label4
            // 
            label4.AutoSize = true;
            label4.Location = new Point(82, 302);
            label4.Name = "label4";
            label4.Size = new Size(116, 17);
            label4.TabIndex = 1;
            label4.Text = "Round Parameters";
            // 
            // label3
            // 
            label3.AutoSize = true;
            label3.Location = new Point(298, 19);
            label3.Name = "label3";
            label3.Size = new Size(204, 17);
            label3.TabIndex = 0;
            label3.Text = "Reagent Round Volume and Time";
            // 
            // Tipbox_Reload
            // 
            Tipbox_Reload.Location = new Point(423, 57);
            Tipbox_Reload.Name = "Tipbox_Reload";
            Tipbox_Reload.Size = new Size(120, 23);
            Tipbox_Reload.TabIndex = 64;
            Tipbox_Reload.Text = "Reload Tipbox";
            Tipbox_Reload.UseVisualStyleBackColor = true;
            Tipbox_Reload.Click += Tipbox_Reload_Click;
            // 
            // Substrate_Reload
            // 
            Substrate_Reload.Location = new Point(423, 86);
            Substrate_Reload.Name = "Substrate_Reload";
            Substrate_Reload.Size = new Size(120, 23);
            Substrate_Reload.TabIndex = 65;
            Substrate_Reload.Text = "Reload Substrate";
            Substrate_Reload.UseVisualStyleBackColor = true;
            Substrate_Reload.Click += Substrate_Reload_Click;
            // 
            // OpenSpecForm
            // 
            OpenSpecForm.Location = new Point(824, 12);
            OpenSpecForm.Name = "OpenSpecForm";
            OpenSpecForm.Size = new Size(95, 23);
            OpenSpecForm.TabIndex = 66;
            OpenSpecForm.Text = "spectrometer";
            OpenSpecForm.UseVisualStyleBackColor = true;
            OpenSpecForm.Click += OpenSpecForm_Click;
            // 
            // AI_Agent
            // 
            AI_Agent.BackColor = Color.Red;
            AI_Agent.Location = new Point(399, 183);
            AI_Agent.Name = "AI_Agent";
            AI_Agent.Size = new Size(125, 23);
            AI_Agent.TabIndex = 67;
            AI_Agent.Text = "ai control off";
            AI_Agent.UseVisualStyleBackColor = false;
            AI_Agent.Click += AI_Agent_Click;
            // 
            // ClearConf
            // 
            ClearConf.AutoSize = true;
            ClearConf.Location = new Point(443, 166);
            ClearConf.Name = "ClearConf";
            ClearConf.Size = new Size(0, 17);
            ClearConf.TabIndex = 69;
            // 
            // ClearReagent
            // 
            ClearReagent.AccessibleRole = AccessibleRole.None;
            ClearReagent.Location = new Point(355, 203);
            ClearReagent.Name = "ClearReagent";
            ClearReagent.Size = new Size(88, 23);
            ClearReagent.TabIndex = 68;
            ClearReagent.Text = "clear";
            ClearReagent.UseVisualStyleBackColor = true;
            ClearReagent.Click += ClearReagent_Click;
            // 
            // label5
            // 
            label5.AutoSize = true;
            label5.Location = new Point(306, 183);
            label5.Name = "label5";
            label5.Size = new Size(196, 17);
            label5.TabIndex = 70;
            label5.Text = "Clear all Reagent Configurations";
            // 
            // MainForm
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(1162, 758);
            Controls.Add(AI_Agent);
            Controls.Add(OpenSpecForm);
            Controls.Add(Substrate_Reload);
            Controls.Add(Tipbox_Reload);
            Controls.Add(ReagentLayout);
            Controls.Add(label2);
            Controls.Add(label1);
            Controls.Add(SheetName);
            Controls.Add(Refresh);
            Controls.Add(MoveTest);
            Controls.Add(FilePath);
            Controls.Add(OpenDispenserForm);
            Controls.Add(AutoRead);
            Controls.Add(OpenCoaterForm);
            Controls.Add(OpenArmForm);
            Controls.Add(ControlPanel);
            Controls.Add(Response);
            Controls.Add(Method);
            Margin = new Padding(2, 3, 2, 3);
            Name = "MainForm";
            Text = "MainForm";
            Load += MainForm_Load;
            Method.ResumeLayout(false);
            Method.PerformLayout();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion
        private TextBox Response;
        private System.Windows.Forms.Timer SpinTimer;
        private CheckBox AutoRead;
        private Panel ControlPanel;
        private Button OpenArmForm;
        private Button OpenCoaterForm;
        private Button OpenDispenserForm;
        private Button MoveTest;
        private Button Refresh;
        private TextBox FilePath;
        private TextBox SheetName;
        private Label label1;
        private Label label2;
        private TableLayoutPanel ReagentLayout;
        private ListBox ExperimentParameters;
        private ListBox ReagentFeatures;
        private GroupBox Method;
        private Label label4;
        private Label label3;
        private RichTextBox Log;
        private Button Tipbox_Reload;
        private Button Substrate_Reload;
        private Button OpenSpecForm;
        private Label label6;
        private TextBox NameReagent;
        private Button AI_Agent;
        private Label label5;
        private Label ClearConf;
        private Button ClearReagent;
    }
}
