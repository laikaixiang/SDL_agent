namespace CSharpTcpDemo
{
    partial class ArmForm
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
            label1 = new Label();
            textBoxIP = new TextBox();
            label2 = new Label();
            textBoxDashboardPort = new TextBox();
            textBoxMovePort = new TextBox();
            label3 = new Label();
            label4 = new Label();
            textBoxFeedbackPort = new TextBox();
            btnConnect = new Button();
            groupBoxConnect = new GroupBox();
            groupBox2 = new GroupBox();
            cboStatus = new ComboBox();
            textBoxSpeedRatio = new TextBox();
            label6 = new Label();
            btnDOInput = new Button();
            btnSpeedConfirm = new Button();
            btnClearError = new Button();
            btnEnableAgain = new Button();
            btnResetRobot = new Button();
            btnEnable = new Button();
            label5 = new Label();
            label19 = new Label();
            label20 = new Label();
            label21 = new Label();
            textBoxIdx = new TextBox();
            groupBox4 = new GroupBox();
            btnJointMovJ = new Button();
            btnMovL = new Button();
            btnMovJ = new Button();
            textBoxJ4 = new TextBox();
            textBoxRx = new TextBox();
            textBoxJ3 = new TextBox();
            textBoxZ = new TextBox();
            textBoxJ2 = new TextBox();
            textBoxY = new TextBox();
            textBoxJ1 = new TextBox();
            textBoxX = new TextBox();
            label16 = new Label();
            label10 = new Label();
            label15 = new Label();
            label9 = new Label();
            label14 = new Label();
            label13 = new Label();
            label8 = new Label();
            label7 = new Label();
            btnClearErrorInfo = new Button();
            richTextBoxErrInfo = new RichTextBox();
            label26 = new Label();
            labJ4 = new Label();
            labRx = new Label();
            labJ3 = new Label();
            labZ = new Label();
            labJ2 = new Label();
            labY = new Label();
            labJ1 = new Label();
            labX = new Label();
            richTextBoxLog = new RichTextBox();
            groupBoxLog = new GroupBox();
            groupBox1 = new GroupBox();
            labRun = new Label();
            label28 = new Label();
            label27 = new Label();
            label25 = new Label();
            label24 = new Label();
            label23 = new Label();
            label22 = new Label();
            label18 = new Label();
            label17 = new Label();
            label12 = new Label();
            label11 = new Label();
            groupBoxConnect.SuspendLayout();
            groupBox2.SuspendLayout();
            groupBox4.SuspendLayout();
            groupBoxLog.SuspendLayout();
            groupBox1.SuspendLayout();
            SuspendLayout();
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(19, 20);
            label1.Margin = new Padding(4, 0, 4, 0);
            label1.Name = "label1";
            label1.Size = new Size(74, 17);
            label1.TabIndex = 0;
            label1.Text = "IP Address:";
            label1.TextAlign = ContentAlignment.MiddleRight;
            // 
            // textBoxIP
            // 
            textBoxIP.Location = new Point(125, 17);
            textBoxIP.Margin = new Padding(4);
            textBoxIP.Name = "textBoxIP";
            textBoxIP.Size = new Size(124, 23);
            textBoxIP.TabIndex = 1;
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.Location = new Point(249, 20);
            label2.Margin = new Padding(4, 0, 4, 0);
            label2.Name = "label2";
            label2.Size = new Size(76, 17);
            label2.TabIndex = 2;
            label2.Text = "Dashboard:";
            label2.TextAlign = ContentAlignment.MiddleRight;
            // 
            // textBoxDashboardPort
            // 
            textBoxDashboardPort.Location = new Point(342, 17);
            textBoxDashboardPort.Margin = new Padding(4);
            textBoxDashboardPort.Name = "textBoxDashboardPort";
            textBoxDashboardPort.Size = new Size(75, 23);
            textBoxDashboardPort.TabIndex = 3;
            // 
            // textBoxMovePort
            // 
            textBoxMovePort.Location = new Point(233, 56);
            textBoxMovePort.Margin = new Padding(4);
            textBoxMovePort.Name = "textBoxMovePort";
            textBoxMovePort.Size = new Size(74, 23);
            textBoxMovePort.TabIndex = 5;
            // 
            // label3
            // 
            label3.AutoSize = true;
            label3.Location = new Point(172, 59);
            label3.Margin = new Padding(4, 0, 4, 0);
            label3.Name = "label3";
            label3.Size = new Size(44, 17);
            label3.TabIndex = 4;
            label3.Text = "Move:";
            label3.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label4
            // 
            label4.AutoSize = true;
            label4.Location = new Point(14, 59);
            label4.Margin = new Padding(4, 0, 4, 0);
            label4.Name = "label4";
            label4.Size = new Size(67, 17);
            label4.TabIndex = 4;
            label4.Text = "Feedback:";
            label4.TextAlign = ContentAlignment.MiddleRight;
            // 
            // textBoxFeedbackPort
            // 
            textBoxFeedbackPort.Location = new Point(100, 56);
            textBoxFeedbackPort.Margin = new Padding(4);
            textBoxFeedbackPort.Name = "textBoxFeedbackPort";
            textBoxFeedbackPort.Size = new Size(71, 23);
            textBoxFeedbackPort.TabIndex = 5;
            // 
            // btnConnect
            // 
            btnConnect.Location = new Point(342, 51);
            btnConnect.Margin = new Padding(4);
            btnConnect.Name = "btnConnect";
            btnConnect.Size = new Size(139, 33);
            btnConnect.TabIndex = 6;
            btnConnect.Text = "Connect";
            btnConnect.UseVisualStyleBackColor = true;
            btnConnect.Click += btnConnect_Click;
            // 
            // groupBoxConnect
            // 
            groupBoxConnect.Controls.Add(label1);
            groupBoxConnect.Controls.Add(btnConnect);
            groupBoxConnect.Controls.Add(textBoxIP);
            groupBoxConnect.Controls.Add(textBoxFeedbackPort);
            groupBoxConnect.Controls.Add(label4);
            groupBoxConnect.Controls.Add(textBoxMovePort);
            groupBoxConnect.Controls.Add(textBoxDashboardPort);
            groupBoxConnect.Controls.Add(label2);
            groupBoxConnect.Controls.Add(label3);
            groupBoxConnect.Location = new Point(14, 13);
            groupBoxConnect.Margin = new Padding(4);
            groupBoxConnect.Name = "groupBoxConnect";
            groupBoxConnect.Padding = new Padding(4);
            groupBoxConnect.Size = new Size(549, 101);
            groupBoxConnect.TabIndex = 7;
            groupBoxConnect.TabStop = false;
            groupBoxConnect.Text = "Robot Connect Ports";
            // 
            // groupBox2
            // 
            groupBox2.Controls.Add(cboStatus);
            groupBox2.Controls.Add(textBoxSpeedRatio);
            groupBox2.Controls.Add(label6);
            groupBox2.Controls.Add(btnDOInput);
            groupBox2.Controls.Add(btnSpeedConfirm);
            groupBox2.Controls.Add(btnClearError);
            groupBox2.Controls.Add(btnEnableAgain);
            groupBox2.Controls.Add(btnResetRobot);
            groupBox2.Controls.Add(btnEnable);
            groupBox2.Controls.Add(label5);
            groupBox2.Controls.Add(label19);
            groupBox2.Controls.Add(label20);
            groupBox2.Controls.Add(label21);
            groupBox2.Controls.Add(textBoxIdx);
            groupBox2.Location = new Point(14, 122);
            groupBox2.Margin = new Padding(4);
            groupBox2.Name = "groupBox2";
            groupBox2.Padding = new Padding(4);
            groupBox2.Size = new Size(680, 156);
            groupBox2.TabIndex = 8;
            groupBox2.TabStop = false;
            groupBox2.Text = "Dashboard Function";
            // 
            // cboStatus
            // 
            cboStatus.DropDownStyle = ComboBoxStyle.DropDownList;
            cboStatus.FormattingEnabled = true;
            cboStatus.Items.AddRange(new object[] { "On", "Off" });
            cboStatus.Location = new Point(179, 105);
            cboStatus.Margin = new Padding(4);
            cboStatus.Name = "cboStatus";
            cboStatus.Size = new Size(67, 25);
            cboStatus.TabIndex = 3;
            // 
            // textBoxSpeedRatio
            // 
            textBoxSpeedRatio.Location = new Point(436, 46);
            textBoxSpeedRatio.Margin = new Padding(4);
            textBoxSpeedRatio.Name = "textBoxSpeedRatio";
            textBoxSpeedRatio.Size = new Size(48, 23);
            textBoxSpeedRatio.TabIndex = 6;
            // 
            // label6
            // 
            label6.AutoSize = true;
            label6.Location = new Point(492, 51);
            label6.Margin = new Padding(4, 0, 4, 0);
            label6.Name = "label6";
            label6.Size = new Size(19, 17);
            label6.TabIndex = 5;
            label6.Text = "%";
            // 
            // btnDOInput
            // 
            btnDOInput.Location = new Point(257, 102);
            btnDOInput.Margin = new Padding(4);
            btnDOInput.Name = "btnDOInput";
            btnDOInput.Size = new Size(88, 33);
            btnDOInput.TabIndex = 0;
            btnDOInput.Text = "Confirm";
            btnDOInput.UseVisualStyleBackColor = true;
            btnDOInput.Click += btnDOInput_Click;
            // 
            // btnSpeedConfirm
            // 
            btnSpeedConfirm.Location = new Point(519, 43);
            btnSpeedConfirm.Margin = new Padding(4);
            btnSpeedConfirm.Name = "btnSpeedConfirm";
            btnSpeedConfirm.Size = new Size(88, 33);
            btnSpeedConfirm.TabIndex = 0;
            btnSpeedConfirm.Text = "Confirm";
            btnSpeedConfirm.UseVisualStyleBackColor = true;
            btnSpeedConfirm.Click += btnSpeedConfirm_Click;
            // 
            // btnClearError
            // 
            btnClearError.Location = new Point(277, 41);
            btnClearError.Margin = new Padding(4);
            btnClearError.Name = "btnClearError";
            btnClearError.Size = new Size(126, 33);
            btnClearError.TabIndex = 0;
            btnClearError.Text = "Clear Error";
            btnClearError.UseVisualStyleBackColor = true;
            btnClearError.Click += btnClearError_Click;
            // 
            // btnEnableAgain
            // 
            btnEnableAgain.Location = new Point(353, 102);
            btnEnableAgain.Margin = new Padding(4);
            btnEnableAgain.Name = "btnEnableAgain";
            btnEnableAgain.Size = new Size(122, 33);
            btnEnableAgain.TabIndex = 0;
            btnEnableAgain.Text = "Enable-Again";
            btnEnableAgain.UseVisualStyleBackColor = true;
            btnEnableAgain.Click += btnEnableAgain_Click;
            // 
            // btnResetRobot
            // 
            btnResetRobot.Location = new Point(147, 41);
            btnResetRobot.Margin = new Padding(4);
            btnResetRobot.Name = "btnResetRobot";
            btnResetRobot.Size = new Size(122, 33);
            btnResetRobot.TabIndex = 0;
            btnResetRobot.Text = "Reset Robot";
            btnResetRobot.UseVisualStyleBackColor = true;
            btnResetRobot.Click += btnResetRobot_Click;
            // 
            // btnEnable
            // 
            btnEnable.Location = new Point(21, 41);
            btnEnable.Margin = new Padding(4);
            btnEnable.Name = "btnEnable";
            btnEnable.Size = new Size(122, 33);
            btnEnable.TabIndex = 0;
            btnEnable.Text = "Enable";
            btnEnable.UseVisualStyleBackColor = true;
            btnEnable.Click += btnEnable_Click;
            // 
            // label5
            // 
            label5.AutoSize = true;
            label5.Location = new Point(419, 20);
            label5.Margin = new Padding(4, 0, 4, 0);
            label5.Name = "label5";
            label5.Size = new Size(82, 17);
            label5.TabIndex = 4;
            label5.Text = "Speed Ratio:";
            label5.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label19
            // 
            label19.AutoSize = true;
            label19.Location = new Point(8, 78);
            label19.Margin = new Padding(4, 0, 4, 0);
            label19.Name = "label19";
            label19.Size = new Size(98, 17);
            label19.TabIndex = 0;
            label19.Text = "Digital Outputs:";
            label19.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label20
            // 
            label20.AutoSize = true;
            label20.Location = new Point(14, 110);
            label20.Margin = new Padding(4, 0, 4, 0);
            label20.Name = "label20";
            label20.Size = new Size(43, 17);
            label20.TabIndex = 0;
            label20.Text = "Index:";
            label20.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label21
            // 
            label21.AutoSize = true;
            label21.Location = new Point(125, 110);
            label21.Margin = new Padding(4, 0, 4, 0);
            label21.Name = "label21";
            label21.Size = new Size(46, 17);
            label21.TabIndex = 0;
            label21.Text = "Status:";
            label21.TextAlign = ContentAlignment.MiddleRight;
            // 
            // textBoxIdx
            // 
            textBoxIdx.Location = new Point(65, 105);
            textBoxIdx.Margin = new Padding(4);
            textBoxIdx.Name = "textBoxIdx";
            textBoxIdx.Size = new Size(59, 23);
            textBoxIdx.TabIndex = 1;
            // 
            // groupBox4
            // 
            groupBox4.Controls.Add(btnJointMovJ);
            groupBox4.Controls.Add(btnMovL);
            groupBox4.Controls.Add(btnMovJ);
            groupBox4.Controls.Add(textBoxJ4);
            groupBox4.Controls.Add(textBoxRx);
            groupBox4.Controls.Add(textBoxJ3);
            groupBox4.Controls.Add(textBoxZ);
            groupBox4.Controls.Add(textBoxJ2);
            groupBox4.Controls.Add(textBoxY);
            groupBox4.Controls.Add(textBoxJ1);
            groupBox4.Controls.Add(textBoxX);
            groupBox4.Controls.Add(label16);
            groupBox4.Controls.Add(label10);
            groupBox4.Controls.Add(label15);
            groupBox4.Controls.Add(label9);
            groupBox4.Controls.Add(label14);
            groupBox4.Controls.Add(label13);
            groupBox4.Controls.Add(label8);
            groupBox4.Controls.Add(label7);
            groupBox4.Location = new Point(14, 286);
            groupBox4.Margin = new Padding(4);
            groupBox4.Name = "groupBox4";
            groupBox4.Padding = new Padding(4);
            groupBox4.Size = new Size(714, 129);
            groupBox4.TabIndex = 8;
            groupBox4.TabStop = false;
            groupBox4.Text = "Move Function";
            // 
            // btnJointMovJ
            // 
            btnJointMovJ.Location = new Point(501, 85);
            btnJointMovJ.Margin = new Padding(4);
            btnJointMovJ.Name = "btnJointMovJ";
            btnJointMovJ.Size = new Size(88, 33);
            btnJointMovJ.TabIndex = 2;
            btnJointMovJ.Text = "JointMovJ";
            btnJointMovJ.UseVisualStyleBackColor = true;
            btnJointMovJ.Click += btnJointMovJ_Click;
            // 
            // btnMovL
            // 
            btnMovL.Location = new Point(602, 35);
            btnMovL.Margin = new Padding(4);
            btnMovL.Name = "btnMovL";
            btnMovL.Size = new Size(88, 33);
            btnMovL.TabIndex = 2;
            btnMovL.Text = "MovL";
            btnMovL.UseVisualStyleBackColor = true;
            btnMovL.Click += btnMovL_Click;
            // 
            // btnMovJ
            // 
            btnMovJ.Location = new Point(501, 35);
            btnMovJ.Margin = new Padding(4);
            btnMovJ.Name = "btnMovJ";
            btnMovJ.Size = new Size(88, 33);
            btnMovJ.TabIndex = 2;
            btnMovJ.Text = "MovJ";
            btnMovJ.UseVisualStyleBackColor = true;
            btnMovJ.Click += btnMovJ_Click;
            // 
            // textBoxJ4
            // 
            textBoxJ4.Location = new Point(398, 90);
            textBoxJ4.Margin = new Padding(4);
            textBoxJ4.Name = "textBoxJ4";
            textBoxJ4.Size = new Size(83, 23);
            textBoxJ4.TabIndex = 1;
            // 
            // textBoxRx
            // 
            textBoxRx.Location = new Point(398, 38);
            textBoxRx.Margin = new Padding(4);
            textBoxRx.Name = "textBoxRx";
            textBoxRx.Size = new Size(83, 23);
            textBoxRx.TabIndex = 1;
            // 
            // textBoxJ3
            // 
            textBoxJ3.Location = new Point(277, 91);
            textBoxJ3.Margin = new Padding(4);
            textBoxJ3.Name = "textBoxJ3";
            textBoxJ3.Size = new Size(83, 23);
            textBoxJ3.TabIndex = 1;
            // 
            // textBoxZ
            // 
            textBoxZ.Location = new Point(277, 38);
            textBoxZ.Margin = new Padding(4);
            textBoxZ.Name = "textBoxZ";
            textBoxZ.Size = new Size(83, 23);
            textBoxZ.TabIndex = 1;
            // 
            // textBoxJ2
            // 
            textBoxJ2.Location = new Point(159, 90);
            textBoxJ2.Margin = new Padding(4);
            textBoxJ2.Name = "textBoxJ2";
            textBoxJ2.Size = new Size(83, 23);
            textBoxJ2.TabIndex = 1;
            // 
            // textBoxY
            // 
            textBoxY.Location = new Point(158, 38);
            textBoxY.Margin = new Padding(4);
            textBoxY.Name = "textBoxY";
            textBoxY.Size = new Size(83, 23);
            textBoxY.TabIndex = 1;
            // 
            // textBoxJ1
            // 
            textBoxJ1.Location = new Point(41, 90);
            textBoxJ1.Margin = new Padding(4);
            textBoxJ1.Name = "textBoxJ1";
            textBoxJ1.Size = new Size(83, 23);
            textBoxJ1.TabIndex = 1;
            // 
            // textBoxX
            // 
            textBoxX.Location = new Point(41, 38);
            textBoxX.Margin = new Padding(4);
            textBoxX.Name = "textBoxX";
            textBoxX.Size = new Size(83, 23);
            textBoxX.TabIndex = 1;
            // 
            // label16
            // 
            label16.AutoSize = true;
            label16.Location = new Point(373, 94);
            label16.Margin = new Padding(4, 0, 4, 0);
            label16.Name = "label16";
            label16.Size = new Size(23, 17);
            label16.TabIndex = 0;
            label16.Text = "J4:";
            label16.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label10
            // 
            label10.AutoSize = true;
            label10.Location = new Point(373, 41);
            label10.Margin = new Padding(4, 0, 4, 0);
            label10.Name = "label10";
            label10.Size = new Size(19, 17);
            label10.TabIndex = 0;
            label10.Text = "R:";
            label10.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label15
            // 
            label15.AutoSize = true;
            label15.Location = new Point(249, 94);
            label15.Margin = new Padding(4, 0, 4, 0);
            label15.Name = "label15";
            label15.Size = new Size(23, 17);
            label15.TabIndex = 0;
            label15.Text = "J3:";
            label15.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label9
            // 
            label9.AutoSize = true;
            label9.Location = new Point(254, 41);
            label9.Margin = new Padding(4, 0, 4, 0);
            label9.Name = "label9";
            label9.Size = new Size(18, 17);
            label9.TabIndex = 0;
            label9.Text = "Z:";
            label9.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label14
            // 
            label14.AutoSize = true;
            label14.Location = new Point(132, 94);
            label14.Margin = new Padding(4, 0, 4, 0);
            label14.Name = "label14";
            label14.Size = new Size(23, 17);
            label14.TabIndex = 0;
            label14.Text = "J2:";
            label14.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label13
            // 
            label13.AutoSize = true;
            label13.Location = new Point(14, 94);
            label13.Margin = new Padding(4, 0, 4, 0);
            label13.Name = "label13";
            label13.Size = new Size(23, 17);
            label13.TabIndex = 0;
            label13.Text = "J1:";
            label13.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label8
            // 
            label8.AutoSize = true;
            label8.Location = new Point(132, 41);
            label8.Margin = new Padding(4, 0, 4, 0);
            label8.Name = "label8";
            label8.Size = new Size(18, 17);
            label8.TabIndex = 0;
            label8.Text = "Y:";
            label8.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label7
            // 
            label7.AutoSize = true;
            label7.Location = new Point(14, 41);
            label7.Margin = new Padding(4, 0, 4, 0);
            label7.Name = "label7";
            label7.Size = new Size(19, 17);
            label7.TabIndex = 0;
            label7.Text = "X:";
            label7.TextAlign = ContentAlignment.MiddleRight;
            // 
            // btnClearErrorInfo
            // 
            btnClearErrorInfo.Location = new Point(24, 253);
            btnClearErrorInfo.Margin = new Padding(4);
            btnClearErrorInfo.Name = "btnClearErrorInfo";
            btnClearErrorInfo.Size = new Size(69, 33);
            btnClearErrorInfo.TabIndex = 4;
            btnClearErrorInfo.Text = "Clear";
            btnClearErrorInfo.UseVisualStyleBackColor = true;
            btnClearErrorInfo.Click += btnClearErrorInfo_Click;
            // 
            // richTextBoxErrInfo
            // 
            richTextBoxErrInfo.Location = new Point(159, 159);
            richTextBoxErrInfo.Margin = new Padding(4);
            richTextBoxErrInfo.Name = "richTextBoxErrInfo";
            richTextBoxErrInfo.Size = new Size(183, 215);
            richTextBoxErrInfo.TabIndex = 2;
            richTextBoxErrInfo.Text = "";
            // 
            // label26
            // 
            label26.AutoSize = true;
            label26.Location = new Point(520, 513);
            label26.Margin = new Padding(4, 0, 4, 0);
            label26.Name = "label26";
            label26.Size = new Size(65, 17);
            label26.TabIndex = 1;
            label26.Text = "Error Info";
            // 
            // labJ4
            // 
            labJ4.Location = new Point(80, 115);
            labJ4.Margin = new Padding(4, 0, 4, 0);
            labJ4.Name = "labJ4";
            labJ4.Size = new Size(91, 20);
            labJ4.TabIndex = 0;
            labJ4.Text = "0.000";
            labJ4.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // labRx
            // 
            labRx.Location = new Point(246, 118);
            labRx.Margin = new Padding(4, 0, 4, 0);
            labRx.Name = "labRx";
            labRx.Size = new Size(93, 17);
            labRx.TabIndex = 0;
            labRx.Text = "0.000";
            labRx.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // labJ3
            // 
            labJ3.Location = new Point(80, 92);
            labJ3.Margin = new Padding(4, 0, 4, 0);
            labJ3.Name = "labJ3";
            labJ3.Size = new Size(92, 17);
            labJ3.TabIndex = 0;
            labJ3.Text = "0.000";
            labJ3.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // labZ
            // 
            labZ.Location = new Point(244, 90);
            labZ.Margin = new Padding(4, 0, 4, 0);
            labZ.Name = "labZ";
            labZ.Size = new Size(100, 22);
            labZ.TabIndex = 0;
            labZ.Text = "0.000";
            labZ.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // labJ2
            // 
            labJ2.Location = new Point(80, 66);
            labJ2.Margin = new Padding(4, 0, 4, 0);
            labJ2.Name = "labJ2";
            labJ2.Size = new Size(91, 22);
            labJ2.TabIndex = 0;
            labJ2.Text = "0.000";
            labJ2.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // labY
            // 
            labY.Location = new Point(245, 67);
            labY.Margin = new Padding(4, 0, 4, 0);
            labY.Name = "labY";
            labY.Size = new Size(100, 21);
            labY.TabIndex = 0;
            labY.Text = "0.000";
            labY.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // labJ1
            // 
            labJ1.Location = new Point(91, 46);
            labJ1.Margin = new Padding(4, 0, 4, 0);
            labJ1.Name = "labJ1";
            labJ1.Size = new Size(69, 17);
            labJ1.TabIndex = 0;
            labJ1.Text = "0.000";
            labJ1.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // labX
            // 
            labX.Location = new Point(254, 42);
            labX.Margin = new Padding(4, 0, 4, 0);
            labX.Name = "labX";
            labX.Size = new Size(85, 24);
            labX.TabIndex = 0;
            labX.Text = "0.000";
            labX.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // richTextBoxLog
            // 
            richTextBoxLog.Location = new Point(9, 28);
            richTextBoxLog.Margin = new Padding(4);
            richTextBoxLog.Name = "richTextBoxLog";
            richTextBoxLog.Size = new Size(266, 347);
            richTextBoxLog.TabIndex = 2;
            richTextBoxLog.Text = "";
            // 
            // groupBoxLog
            // 
            groupBoxLog.Controls.Add(richTextBoxLog);
            groupBoxLog.Location = new Point(386, 423);
            groupBoxLog.Margin = new Padding(4);
            groupBoxLog.Name = "groupBoxLog";
            groupBoxLog.Padding = new Padding(4);
            groupBoxLog.Size = new Size(284, 383);
            groupBoxLog.TabIndex = 10;
            groupBoxLog.TabStop = false;
            groupBoxLog.Text = "Log";
            // 
            // groupBox1
            // 
            groupBox1.Controls.Add(labRun);
            groupBox1.Controls.Add(label28);
            groupBox1.Controls.Add(label27);
            groupBox1.Controls.Add(label25);
            groupBox1.Controls.Add(label24);
            groupBox1.Controls.Add(label23);
            groupBox1.Controls.Add(label22);
            groupBox1.Controls.Add(label18);
            groupBox1.Controls.Add(label17);
            groupBox1.Controls.Add(label12);
            groupBox1.Controls.Add(btnClearErrorInfo);
            groupBox1.Controls.Add(label11);
            groupBox1.Controls.Add(labJ1);
            groupBox1.Controls.Add(labJ2);
            groupBox1.Controls.Add(labJ3);
            groupBox1.Controls.Add(labJ4);
            groupBox1.Controls.Add(labX);
            groupBox1.Controls.Add(labRx);
            groupBox1.Controls.Add(richTextBoxErrInfo);
            groupBox1.Controls.Add(labZ);
            groupBox1.Controls.Add(labY);
            groupBox1.Location = new Point(14, 423);
            groupBox1.Margin = new Padding(4);
            groupBox1.Name = "groupBox1";
            groupBox1.Padding = new Padding(4);
            groupBox1.Size = new Size(352, 383);
            groupBox1.TabIndex = 11;
            groupBox1.TabStop = false;
            groupBox1.Text = "Position";
            // 
            // labRun
            // 
            labRun.AutoSize = true;
            labRun.Location = new Point(91, 159);
            labRun.Name = "labRun";
            labRun.Size = new Size(15, 17);
            labRun.TabIndex = 12;
            labRun.Text = "0";
            // 
            // label28
            // 
            label28.AutoSize = true;
            label28.Location = new Point(14, 159);
            label28.Name = "label28";
            label28.Size = new Size(58, 17);
            label28.TabIndex = 21;
            label28.Text = "Running:";
            // 
            // label27
            // 
            label27.AutoSize = true;
            label27.Location = new Point(233, 118);
            label27.Name = "label27";
            label27.Size = new Size(19, 17);
            label27.TabIndex = 20;
            label27.Text = "R:";
            // 
            // label25
            // 
            label25.AutoSize = true;
            label25.Location = new Point(233, 70);
            label25.Name = "label25";
            label25.Size = new Size(18, 17);
            label25.TabIndex = 19;
            label25.Text = "Y:";
            // 
            // label24
            // 
            label24.AutoSize = true;
            label24.Location = new Point(233, 92);
            label24.Name = "label24";
            label24.Size = new Size(18, 17);
            label24.TabIndex = 18;
            label24.Text = "Z:";
            // 
            // label23
            // 
            label23.AutoSize = true;
            label23.Location = new Point(60, 69);
            label23.Name = "label23";
            label23.Size = new Size(23, 17);
            label23.TabIndex = 17;
            label23.Text = "J2:";
            // 
            // label22
            // 
            label22.AutoSize = true;
            label22.Location = new Point(58, 92);
            label22.Name = "label22";
            label22.Size = new Size(23, 17);
            label22.TabIndex = 16;
            label22.Text = "J3:";
            // 
            // label18
            // 
            label18.AutoSize = true;
            label18.Location = new Point(58, 118);
            label18.Name = "label18";
            label18.Size = new Size(23, 17);
            label18.TabIndex = 15;
            label18.Text = "J4:";
            // 
            // label17
            // 
            label17.AutoSize = true;
            label17.Location = new Point(233, 46);
            label17.Name = "label17";
            label17.Size = new Size(19, 17);
            label17.TabIndex = 14;
            label17.Text = "X:";
            // 
            // label12
            // 
            label12.AutoSize = true;
            label12.Location = new Point(61, 46);
            label12.Name = "label12";
            label12.Size = new Size(23, 17);
            label12.TabIndex = 13;
            label12.Text = "J1:";
            // 
            // label11
            // 
            label11.AutoSize = true;
            label11.Location = new Point(37, 232);
            label11.Margin = new Padding(4, 0, 4, 0);
            label11.Name = "label11";
            label11.Size = new Size(44, 17);
            label11.TabIndex = 12;
            label11.Text = "errors";
            // 
            // ArmForm
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(741, 866);
            Controls.Add(groupBoxLog);
            Controls.Add(groupBox4);
            Controls.Add(groupBox2);
            Controls.Add(groupBoxConnect);
            Controls.Add(label26);
            Controls.Add(groupBox1);
            FormBorderStyle = FormBorderStyle.FixedSingle;
            Margin = new Padding(4);
            MaximizeBox = false;
            Name = "ArmForm";
            Text = "ArmForm";
            FormClosed += ArmForm_FormClosed;
            Load += ArmForm_Load;
            groupBoxConnect.ResumeLayout(false);
            groupBoxConnect.PerformLayout();
            groupBox2.ResumeLayout(false);
            groupBox2.PerformLayout();
            groupBox4.ResumeLayout(false);
            groupBox4.PerformLayout();
            groupBoxLog.ResumeLayout(false);
            groupBox1.ResumeLayout(false);
            groupBox1.PerformLayout();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox textBoxIP;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox textBoxDashboardPort;
        private System.Windows.Forms.TextBox textBoxMovePort;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox textBoxFeedbackPort;
        private System.Windows.Forms.Button btnConnect;
        private System.Windows.Forms.GroupBox groupBoxConnect;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.TextBox textBoxSpeedRatio;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Button btnSpeedConfirm;
        private System.Windows.Forms.Button btnClearError;
        private System.Windows.Forms.Button btnResetRobot;
        private System.Windows.Forms.Button btnEnable;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.GroupBox groupBox4;
        private System.Windows.Forms.TextBox textBoxJ4;
        private System.Windows.Forms.TextBox textBoxRx;
        private System.Windows.Forms.TextBox textBoxJ3;
        private System.Windows.Forms.TextBox textBoxZ;
        private System.Windows.Forms.TextBox textBoxJ2;
        private System.Windows.Forms.TextBox textBoxY;
        private System.Windows.Forms.TextBox textBoxJ1;
        private System.Windows.Forms.TextBox textBoxX;
        private System.Windows.Forms.Label label16;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.ComboBox cboStatus;
        private System.Windows.Forms.Button btnJointMovJ;
        private System.Windows.Forms.Button btnDOInput;
        private System.Windows.Forms.Button btnMovL;
        private System.Windows.Forms.Button btnMovJ;
        private System.Windows.Forms.TextBox textBoxIdx;
        private System.Windows.Forms.Label label21;
        private System.Windows.Forms.Label label20;
        private System.Windows.Forms.Label label19;
        private System.Windows.Forms.RichTextBox richTextBoxErrInfo;
        private System.Windows.Forms.Label label26;
        private System.Windows.Forms.Button btnClearErrorInfo;
        private System.Windows.Forms.Label labJ4;
        private System.Windows.Forms.Label labRx;
        private System.Windows.Forms.Label labJ3;
        private System.Windows.Forms.Label labZ;
        private System.Windows.Forms.Label labJ2;
        private System.Windows.Forms.Label labY;
        private System.Windows.Forms.Label labJ1;
        private System.Windows.Forms.Label labX;
        private System.Windows.Forms.RichTextBox richTextBoxLog;
        private System.Windows.Forms.GroupBox groupBoxLog;
        private System.Windows.Forms.Button btnEnableAgain;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Label label11;
        private Label label27;
        private Label label25;
        private Label label24;
        private Label label23;
        private Label label22;
        private Label label18;
        private Label label17;
        private Label label12;
        private Label labRun;
        private Label label28;
    }
}