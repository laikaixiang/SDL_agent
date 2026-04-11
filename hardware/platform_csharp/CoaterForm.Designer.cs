namespace WinFormsApp_Draft
{
    partial class CoaterForm
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
            SpinCoater = new GroupBox();
            SpinSpeed = new TextBox();
            label2 = new Label();
            SendRequest = new Button();
            ForceStop = new Button();
            FreeStop = new Button();
            AccSpeed = new TextBox();
            label4 = new Label();
            SpinDuration = new TextBox();
            label3 = new Label();
            ClearPos = new Button();
            Update = new CheckBox();
            Position = new TextBox();
            label1 = new Label();
            ResetMotor = new Button();
            MotorConnectionState = new TextBox();
            MotorSerialSwitch = new Button();
            SelectedMotorBR = new TextBox();
            MotorBaudRate = new ListBox();
            SelectedMotorPort = new TextBox();
            MotorPorts = new ListBox();
            Response = new TextBox();
            SpinCoater.SuspendLayout();
            SuspendLayout();
            // 
            // SpinCoater
            // 
            SpinCoater.Controls.Add(SpinSpeed);
            SpinCoater.Controls.Add(label2);
            SpinCoater.Controls.Add(SendRequest);
            SpinCoater.Controls.Add(ForceStop);
            SpinCoater.Controls.Add(FreeStop);
            SpinCoater.Controls.Add(AccSpeed);
            SpinCoater.Controls.Add(label4);
            SpinCoater.Controls.Add(SpinDuration);
            SpinCoater.Controls.Add(label3);
            SpinCoater.Controls.Add(ClearPos);
            SpinCoater.Controls.Add(Update);
            SpinCoater.Controls.Add(Position);
            SpinCoater.Controls.Add(label1);
            SpinCoater.Controls.Add(ResetMotor);
            SpinCoater.Location = new Point(49, 196);
            SpinCoater.Name = "SpinCoater";
            SpinCoater.Size = new Size(385, 188);
            SpinCoater.TabIndex = 53;
            SpinCoater.TabStop = false;
            SpinCoater.Text = "Spin Coater";
            // 
            // SpinSpeed
            // 
            SpinSpeed.Location = new Point(25, 50);
            SpinSpeed.Margin = new Padding(2, 3, 2, 3);
            SpinSpeed.Name = "SpinSpeed";
            SpinSpeed.Size = new Size(98, 23);
            SpinSpeed.TabIndex = 25;
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.Location = new Point(17, 30);
            label2.Margin = new Padding(2, 0, 2, 0);
            label2.Name = "label2";
            label2.Size = new Size(107, 17);
            label2.TabIndex = 19;
            label2.Text = "spin speed(RPM)";
            // 
            // SendRequest
            // 
            SendRequest.Location = new Point(128, 47);
            SendRequest.Margin = new Padding(2, 3, 2, 3);
            SendRequest.Name = "SendRequest";
            SendRequest.Size = new Size(73, 25);
            SendRequest.TabIndex = 9;
            SendRequest.Text = "send";
            SendRequest.UseVisualStyleBackColor = true;
            SendRequest.Click += SendRequest_Click;
            // 
            // ForceStop
            // 
            ForceStop.Location = new Point(205, 45);
            ForceStop.Margin = new Padding(2, 3, 2, 3);
            ForceStop.Name = "ForceStop";
            ForceStop.Size = new Size(85, 27);
            ForceStop.TabIndex = 11;
            ForceStop.Text = "force stop";
            ForceStop.UseVisualStyleBackColor = true;
            ForceStop.Click += ForceStop_Click;
            // 
            // FreeStop
            // 
            FreeStop.Location = new Point(294, 47);
            FreeStop.Margin = new Padding(2, 3, 2, 3);
            FreeStop.Name = "FreeStop";
            FreeStop.Size = new Size(73, 25);
            FreeStop.TabIndex = 12;
            FreeStop.Text = "free stop";
            FreeStop.UseVisualStyleBackColor = true;
            FreeStop.Click += FreeStop_Click;
            // 
            // AccSpeed
            // 
            AccSpeed.Location = new Point(25, 148);
            AccSpeed.Margin = new Padding(2, 3, 2, 3);
            AccSpeed.Name = "AccSpeed";
            AccSpeed.Size = new Size(96, 23);
            AccSpeed.TabIndex = 22;
            // 
            // label4
            // 
            label4.AutoSize = true;
            label4.Location = new Point(17, 127);
            label4.Margin = new Padding(2, 0, 2, 0);
            label4.Name = "label4";
            label4.Size = new Size(113, 17);
            label4.TabIndex = 23;
            label4.Text = "acc speed(RPM/s)";
            // 
            // SpinDuration
            // 
            SpinDuration.Location = new Point(25, 101);
            SpinDuration.Margin = new Padding(2, 3, 2, 3);
            SpinDuration.Name = "SpinDuration";
            SpinDuration.Size = new Size(96, 23);
            SpinDuration.TabIndex = 20;
            // 
            // label3
            // 
            label3.AutoSize = true;
            label3.Location = new Point(22, 81);
            label3.Margin = new Padding(2, 0, 2, 0);
            label3.Name = "label3";
            label3.Size = new Size(99, 17);
            label3.TabIndex = 21;
            label3.Text = "spin duration(s)";
            // 
            // ClearPos
            // 
            ClearPos.Location = new Point(166, 145);
            ClearPos.Margin = new Padding(2, 3, 2, 3);
            ClearPos.Name = "ClearPos";
            ClearPos.Size = new Size(73, 25);
            ClearPos.TabIndex = 15;
            ClearPos.Text = "clear";
            ClearPos.TextImageRelation = TextImageRelation.ImageAboveText;
            ClearPos.UseVisualStyleBackColor = true;
            ClearPos.Click += ClearPos_Click;
            // 
            // Update
            // 
            Update.AutoSize = true;
            Update.Location = new Point(166, 118);
            Update.Margin = new Padding(2, 3, 2, 3);
            Update.Name = "Update";
            Update.Size = new Size(68, 21);
            Update.TabIndex = 16;
            Update.Text = "update";
            Update.UseVisualStyleBackColor = true;
            Update.CheckedChanged += Update_CheckedChanged;
            // 
            // Position
            // 
            Position.Location = new Point(146, 95);
            Position.Margin = new Padding(2, 3, 2, 3);
            Position.Name = "Position";
            Position.ReadOnly = true;
            Position.Size = new Size(100, 23);
            Position.TabIndex = 13;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(142, 75);
            label1.Margin = new Padding(2, 0, 2, 0);
            label1.Name = "label1";
            label1.Size = new Size(104, 17);
            label1.TabIndex = 14;
            label1.Text = "position: degree";
            // 
            // ResetMotor
            // 
            ResetMotor.Location = new Point(259, 96);
            ResetMotor.Margin = new Padding(2, 3, 2, 3);
            ResetMotor.Name = "ResetMotor";
            ResetMotor.Size = new Size(73, 43);
            ResetMotor.TabIndex = 18;
            ResetMotor.Text = "reset motor";
            ResetMotor.UseVisualStyleBackColor = true;
            ResetMotor.Click += ResetMotor_Click;
            // 
            // MotorConnectionState
            // 
            MotorConnectionState.AccessibleRole = AccessibleRole.WhiteSpace;
            MotorConnectionState.Location = new Point(274, 33);
            MotorConnectionState.Margin = new Padding(2, 3, 2, 3);
            MotorConnectionState.Name = "MotorConnectionState";
            MotorConnectionState.ReadOnly = true;
            MotorConnectionState.Size = new Size(82, 23);
            MotorConnectionState.TabIndex = 52;
            MotorConnectionState.Text = "closed";
            // 
            // MotorSerialSwitch
            // 
            MotorSerialSwitch.Location = new Point(84, 33);
            MotorSerialSwitch.Margin = new Padding(2, 3, 2, 3);
            MotorSerialSwitch.Name = "MotorSerialSwitch";
            MotorSerialSwitch.Size = new Size(110, 25);
            MotorSerialSwitch.TabIndex = 51;
            MotorSerialSwitch.Text = "Connect Motor";
            MotorSerialSwitch.UseVisualStyleBackColor = true;
            MotorSerialSwitch.Click += MotorSerialSwitch_Click;
            // 
            // SelectedMotorBR
            // 
            SelectedMotorBR.Location = new Point(258, 71);
            SelectedMotorBR.Margin = new Padding(2, 3, 2, 3);
            SelectedMotorBR.Name = "SelectedMotorBR";
            SelectedMotorBR.ReadOnly = true;
            SelectedMotorBR.Size = new Size(98, 23);
            SelectedMotorBR.TabIndex = 50;
            // 
            // MotorBaudRate
            // 
            MotorBaudRate.FormattingEnabled = true;
            MotorBaudRate.ItemHeight = 17;
            MotorBaudRate.Location = new Point(258, 110);
            MotorBaudRate.Margin = new Padding(2, 3, 2, 3);
            MotorBaudRate.Name = "MotorBaudRate";
            MotorBaudRate.Size = new Size(98, 55);
            MotorBaudRate.TabIndex = 49;
            MotorBaudRate.SelectedIndexChanged += MotorBaudRate_SelectedIndexChanged;
            // 
            // SelectedMotorPort
            // 
            SelectedMotorPort.Location = new Point(96, 71);
            SelectedMotorPort.Margin = new Padding(2, 3, 2, 3);
            SelectedMotorPort.Name = "SelectedMotorPort";
            SelectedMotorPort.ReadOnly = true;
            SelectedMotorPort.Size = new Size(98, 23);
            SelectedMotorPort.TabIndex = 48;
            // 
            // MotorPorts
            // 
            MotorPorts.FormattingEnabled = true;
            MotorPorts.ItemHeight = 17;
            MotorPorts.Location = new Point(96, 110);
            MotorPorts.Margin = new Padding(2, 3, 2, 3);
            MotorPorts.Name = "MotorPorts";
            MotorPorts.Size = new Size(96, 55);
            MotorPorts.TabIndex = 47;
            MotorPorts.SelectedIndexChanged += MotorPorts_SelectedIndexChanged;
            // 
            // Response
            // 
            Response.Location = new Point(66, 404);
            Response.Name = "Response";
            Response.ReadOnly = true;
            Response.Size = new Size(350, 23);
            Response.TabIndex = 54;
            // 
            // CoaterForm
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(521, 452);
            Controls.Add(Response);
            Controls.Add(SpinCoater);
            Controls.Add(MotorConnectionState);
            Controls.Add(MotorSerialSwitch);
            Controls.Add(SelectedMotorBR);
            Controls.Add(MotorBaudRate);
            Controls.Add(SelectedMotorPort);
            Controls.Add(MotorPorts);
            Name = "CoaterForm";
            Text = "CoaterForm";
            Load += CoaterForm_Load;
            SpinCoater.ResumeLayout(false);
            SpinCoater.PerformLayout();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private GroupBox SpinCoater;
        private TextBox SpinSpeed;
        private Label label2;
        private Button SendRequest;
        private Button ForceStop;
        private Button FreeStop;
        private TextBox AccSpeed;
        private Label label4;
        private TextBox SpinDuration;
        private Label label3;
        private Button ClearPos;
        private CheckBox Update;
        private TextBox Position;
        private Label label1;
        private Button ResetMotor;
        private TextBox MotorConnectionState;
        private Button MotorSerialSwitch;
        private TextBox SelectedMotorBR;
        private ListBox MotorBaudRate;
        private TextBox SelectedMotorPort;
        private ListBox MotorPorts;
        private TextBox Response;
    }
}