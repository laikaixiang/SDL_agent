namespace WinFormsApp_Draft
{
    partial class DispenserForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
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
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            DispenserSerialSwitch = new Button();
            Response = new TextBox();
            DispenserConnectionState = new TextBox();
            ResetDispensor = new Button();
            AxesManeuver_front = new Button();
            X = new TextBox();
            Y = new TextBox();
            LeftZ = new TextBox();
            RightZ = new TextBox();
            label1 = new Label();
            label2 = new Label();
            label3 = new Label();
            label4 = new Label();
            LeftTipSuck = new Button();
            LeftTipSpit = new Button();
            LeftTipEnable = new GroupBox();
            LeftTipDrop = new Button();
            LeftTipVolume = new TextBox();
            label5 = new Label();
            RightTipEnable = new GroupBox();
            RightTipDrop = new Button();
            RightTipVolume = new TextBox();
            label7 = new Label();
            RightTipSuck = new Button();
            RightTipSpit = new Button();
            CheckTip = new Button();
            DispenserPorts = new ListBox();
            AxesManeuver_back = new Button();
            LeftTipEnable.SuspendLayout();
            RightTipEnable.SuspendLayout();
            SuspendLayout();
            // 
            // DispenserSerialSwitch
            // 
            DispenserSerialSwitch.Location = new Point(55, 30);
            DispenserSerialSwitch.Name = "DispenserSerialSwitch";
            DispenserSerialSwitch.Size = new Size(126, 23);
            DispenserSerialSwitch.TabIndex = 0;
            DispenserSerialSwitch.Text = "connect";
            DispenserSerialSwitch.UseVisualStyleBackColor = true;
            DispenserSerialSwitch.Click += DispenserSerialSwitch_Click;
            // 
            // Response
            // 
            Response.Location = new Point(82, 467);
            Response.Name = "Response";
            Response.ReadOnly = true;
            Response.Size = new Size(269, 23);
            Response.TabIndex = 1;
            // 
            // DispenserConnectionState
            // 
            DispenserConnectionState.Location = new Point(47, 59);
            DispenserConnectionState.Name = "DispenserConnectionState";
            DispenserConnectionState.ReadOnly = true;
            DispenserConnectionState.Size = new Size(150, 23);
            DispenserConnectionState.TabIndex = 2;
            DispenserConnectionState.Text = "dispenser disconnected";
            // 
            // ResetDispensor
            // 
            ResetDispensor.Location = new Point(357, 195);
            ResetDispensor.Name = "ResetDispensor";
            ResetDispensor.Size = new Size(75, 23);
            ResetDispensor.TabIndex = 3;
            ResetDispensor.Text = "reset";
            ResetDispensor.UseVisualStyleBackColor = true;
            ResetDispensor.Click += ResetDispensor_Click;
            // 
            // AxesManeuver_front
            // 
            AxesManeuver_front.Location = new Point(292, 133);
            AxesManeuver_front.Name = "AxesManeuver_front";
            AxesManeuver_front.Size = new Size(96, 23);
            AxesManeuver_front.TabIndex = 4;
            AxesManeuver_front.Text = "run_forward";
            AxesManeuver_front.UseVisualStyleBackColor = true;
            AxesManeuver_front.Click += AxesManeuver_front_Click;
            // 
            // X
            // 
            X.Location = new Point(47, 133);
            X.Name = "X";
            X.Size = new Size(83, 23);
            X.TabIndex = 5;
            // 
            // Y
            // 
            Y.Location = new Point(154, 133);
            Y.Name = "Y";
            Y.Size = new Size(83, 23);
            Y.TabIndex = 6;
            // 
            // LeftZ
            // 
            LeftZ.Location = new Point(47, 195);
            LeftZ.Name = "LeftZ";
            LeftZ.Size = new Size(83, 23);
            LeftZ.TabIndex = 7;
            // 
            // RightZ
            // 
            RightZ.Location = new Point(154, 195);
            RightZ.Name = "RightZ";
            RightZ.Size = new Size(83, 23);
            RightZ.TabIndex = 8;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(82, 113);
            label1.Name = "label1";
            label1.Size = new Size(16, 17);
            label1.TabIndex = 9;
            label1.Text = "X";
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.Location = new Point(188, 113);
            label2.Name = "label2";
            label2.Size = new Size(15, 17);
            label2.TabIndex = 10;
            label2.Text = "Y";
            // 
            // label3
            // 
            label3.AutoSize = true;
            label3.Location = new Point(70, 175);
            label3.Name = "label3";
            label3.Size = new Size(40, 17);
            label3.TabIndex = 11;
            label3.Text = "Left Z";
            // 
            // label4
            // 
            label4.AutoSize = true;
            label4.Location = new Point(172, 175);
            label4.Name = "label4";
            label4.Size = new Size(49, 17);
            label4.TabIndex = 12;
            label4.Text = "Right Z";
            // 
            // LeftTipSuck
            // 
            LeftTipSuck.Location = new Point(43, 80);
            LeftTipSuck.Name = "LeftTipSuck";
            LeftTipSuck.Size = new Size(75, 23);
            LeftTipSuck.TabIndex = 13;
            LeftTipSuck.Text = "suck";
            LeftTipSuck.UseVisualStyleBackColor = true;
            LeftTipSuck.Click += LeftTipSuck_Click;
            // 
            // LeftTipSpit
            // 
            LeftTipSpit.Location = new Point(43, 106);
            LeftTipSpit.Name = "LeftTipSpit";
            LeftTipSpit.Size = new Size(75, 23);
            LeftTipSpit.TabIndex = 14;
            LeftTipSpit.Text = "spit";
            LeftTipSpit.UseVisualStyleBackColor = true;
            LeftTipSpit.Click += LeftTipSpit_Click;
            // 
            // LeftTipEnable
            // 
            LeftTipEnable.Controls.Add(LeftTipDrop);
            LeftTipEnable.Controls.Add(LeftTipVolume);
            LeftTipEnable.Controls.Add(label5);
            LeftTipEnable.Controls.Add(LeftTipSuck);
            LeftTipEnable.Controls.Add(LeftTipSpit);
            LeftTipEnable.Location = new Point(12, 250);
            LeftTipEnable.Name = "LeftTipEnable";
            LeftTipEnable.Size = new Size(160, 169);
            LeftTipEnable.TabIndex = 15;
            LeftTipEnable.TabStop = false;
            LeftTipEnable.Text = "LeftTip";
            // 
            // LeftTipDrop
            // 
            LeftTipDrop.Location = new Point(43, 135);
            LeftTipDrop.Name = "LeftTipDrop";
            LeftTipDrop.Size = new Size(75, 23);
            LeftTipDrop.TabIndex = 19;
            LeftTipDrop.Text = "drop";
            LeftTipDrop.UseVisualStyleBackColor = true;
            LeftTipDrop.Click += LeftTipDrop_Click;
            // 
            // LeftTipVolume
            // 
            LeftTipVolume.Location = new Point(35, 51);
            LeftTipVolume.Name = "LeftTipVolume";
            LeftTipVolume.Size = new Size(98, 23);
            LeftTipVolume.TabIndex = 17;
            // 
            // label5
            // 
            label5.AutoSize = true;
            label5.Location = new Point(52, 28);
            label5.Name = "label5";
            label5.Size = new Size(66, 17);
            label5.TabIndex = 15;
            label5.Text = "Left Tip/ul";
            // 
            // RightTipEnable
            // 
            RightTipEnable.Controls.Add(RightTipDrop);
            RightTipEnable.Controls.Add(RightTipVolume);
            RightTipEnable.Controls.Add(label7);
            RightTipEnable.Controls.Add(RightTipSuck);
            RightTipEnable.Controls.Add(RightTipSpit);
            RightTipEnable.Location = new Point(255, 250);
            RightTipEnable.Name = "RightTipEnable";
            RightTipEnable.Size = new Size(160, 169);
            RightTipEnable.TabIndex = 19;
            RightTipEnable.TabStop = false;
            RightTipEnable.Text = "RightTip";
            // 
            // RightTipDrop
            // 
            RightTipDrop.Location = new Point(43, 135);
            RightTipDrop.Name = "RightTipDrop";
            RightTipDrop.Size = new Size(75, 23);
            RightTipDrop.TabIndex = 18;
            RightTipDrop.Text = "drop";
            RightTipDrop.UseVisualStyleBackColor = true;
            RightTipDrop.Click += RightTipDrop_Click;
            // 
            // RightTipVolume
            // 
            RightTipVolume.Location = new Point(35, 51);
            RightTipVolume.Name = "RightTipVolume";
            RightTipVolume.Size = new Size(98, 23);
            RightTipVolume.TabIndex = 17;
            // 
            // label7
            // 
            label7.AutoSize = true;
            label7.Location = new Point(43, 28);
            label7.Name = "label7";
            label7.Size = new Size(75, 17);
            label7.TabIndex = 15;
            label7.Text = "Right Tip/ul";
            // 
            // RightTipSuck
            // 
            RightTipSuck.Location = new Point(43, 80);
            RightTipSuck.Name = "RightTipSuck";
            RightTipSuck.Size = new Size(75, 23);
            RightTipSuck.TabIndex = 13;
            RightTipSuck.Text = "suck";
            RightTipSuck.UseVisualStyleBackColor = true;
            RightTipSuck.Click += RightTipSuck_Click;
            // 
            // RightTipSpit
            // 
            RightTipSpit.Location = new Point(43, 106);
            RightTipSpit.Name = "RightTipSpit";
            RightTipSpit.Size = new Size(75, 23);
            RightTipSpit.TabIndex = 14;
            RightTipSpit.Text = "spit";
            RightTipSpit.UseVisualStyleBackColor = true;
            RightTipSpit.Click += RightTipSpit_Click;
            // 
            // CheckTip
            // 
            CheckTip.Location = new Point(276, 195);
            CheckTip.Name = "CheckTip";
            CheckTip.Size = new Size(75, 23);
            CheckTip.TabIndex = 20;
            CheckTip.Text = "check tip";
            CheckTip.UseVisualStyleBackColor = true;
            CheckTip.Click += CheckTip_Click;
            // 
            // DispenserPorts
            // 
            DispenserPorts.FormattingEnabled = true;
            DispenserPorts.ItemHeight = 17;
            DispenserPorts.Location = new Point(253, 30);
            DispenserPorts.Name = "DispenserPorts";
            DispenserPorts.Size = new Size(120, 89);
            DispenserPorts.TabIndex = 21;
            // 
            // AxesManeuver_back
            // 
            AxesManeuver_back.Location = new Point(290, 162);
            AxesManeuver_back.Name = "AxesManeuver_back";
            AxesManeuver_back.Size = new Size(98, 23);
            AxesManeuver_back.TabIndex = 22;
            AxesManeuver_back.Text = "run_backward";
            AxesManeuver_back.UseVisualStyleBackColor = true;
            AxesManeuver_back.Click += AxesManeuver_back_Click;
            // 
            // DispenserForm
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(444, 516);
            Controls.Add(AxesManeuver_back);
            Controls.Add(DispenserPorts);
            Controls.Add(CheckTip);
            Controls.Add(RightTipEnable);
            Controls.Add(LeftTipEnable);
            Controls.Add(label4);
            Controls.Add(label3);
            Controls.Add(label2);
            Controls.Add(label1);
            Controls.Add(RightZ);
            Controls.Add(LeftZ);
            Controls.Add(Y);
            Controls.Add(X);
            Controls.Add(AxesManeuver_front);
            Controls.Add(ResetDispensor);
            Controls.Add(DispenserConnectionState);
            Controls.Add(Response);
            Controls.Add(DispenserSerialSwitch);
            Name = "DispenserForm";
            Text = "DispenserForm";
            Load += DispenserForm_Load;
            LeftTipEnable.ResumeLayout(false);
            LeftTipEnable.PerformLayout();
            RightTipEnable.ResumeLayout(false);
            RightTipEnable.PerformLayout();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button DispenserSerialSwitch;
        private TextBox Response;
        private TextBox DispenserConnectionState;
        private Button ResetDispensor;
        private Button AxesManeuver_front;
        private TextBox X;
        private TextBox Y;
        private TextBox LeftZ;
        private TextBox RightZ;
        private Label label1;
        private Label label2;
        private Label label3;
        private Label label4;
        private Button LeftTipSuck;
        private Button LeftTipSpit;
        private GroupBox LeftTipEnable;
        private TextBox LeftTipVolume;
        private Label label5;
        private GroupBox RightTipEnable;
        private TextBox RightTipVolume;
        private Label label7;
        private Button RightTipSuck;
        private Button RightTipSpit;
        private Button CheckTip;
        private Button LeftTipDrop;
        private Button RightTipDrop;
        private ListBox DispenserPorts;
        private Button AxesManeuver_back;
    }
}