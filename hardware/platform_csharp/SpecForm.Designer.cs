namespace WinFormsApp_Draft
{
    partial class SpecForm
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
            Connect_Brocker = new Button();
            Broker_addr = new TextBox();
            label1 = new Label();
            Topic = new TextBox();
            label2 = new Label();
            Response0 = new TextBox();
            Publish = new Button();
            Message = new TextBox();
            label4 = new Label();
            Connect_Spectrum = new Button();
            SendProgress = new ProgressBar();
            DisconnectPyClient = new Button();
            SuspendLayout();
            // 
            // Connect_Brocker
            // 
            Connect_Brocker.Location = new Point(37, 230);
            Connect_Brocker.Name = "Connect_Brocker";
            Connect_Brocker.Size = new Size(149, 23);
            Connect_Brocker.TabIndex = 0;
            Connect_Brocker.Text = "connect brocker";
            Connect_Brocker.UseVisualStyleBackColor = true;
            Connect_Brocker.Click += Connect_Brocker_Click;
            // 
            // Broker_addr
            // 
            Broker_addr.Location = new Point(37, 43);
            Broker_addr.Name = "Broker_addr";
            Broker_addr.Size = new Size(149, 23);
            Broker_addr.TabIndex = 1;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(64, 23);
            label1.Name = "label1";
            label1.Size = new Size(99, 17);
            label1.TabIndex = 2;
            label1.Text = "broker address";
            // 
            // Topic
            // 
            Topic.Location = new Point(37, 100);
            Topic.Name = "Topic";
            Topic.Size = new Size(149, 23);
            Topic.TabIndex = 3;
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.Location = new Point(88, 80);
            label2.Name = "label2";
            label2.Size = new Size(37, 17);
            label2.TabIndex = 4;
            label2.Text = "topic";
            // 
            // Response0
            // 
            Response0.Location = new Point(37, 201);
            Response0.Name = "Response0";
            Response0.ReadOnly = true;
            Response0.Size = new Size(149, 23);
            Response0.TabIndex = 5;
            // 
            // Publish
            // 
            Publish.Location = new Point(64, 288);
            Publish.Name = "Publish";
            Publish.Size = new Size(99, 23);
            Publish.TabIndex = 6;
            Publish.Text = "publish";
            Publish.UseVisualStyleBackColor = true;
            Publish.Click += Publish_Click;
            // 
            // Message
            // 
            Message.Location = new Point(37, 157);
            Message.Name = "Message";
            Message.Size = new Size(149, 23);
            Message.TabIndex = 7;
            // 
            // label4
            // 
            label4.AutoSize = true;
            label4.Location = new Point(82, 137);
            label4.Name = "label4";
            label4.Size = new Size(60, 17);
            label4.TabIndex = 9;
            label4.Text = "message";
            // 
            // Connect_Spectrum
            // 
            Connect_Spectrum.Location = new Point(37, 259);
            Connect_Spectrum.Name = "Connect_Spectrum";
            Connect_Spectrum.Size = new Size(149, 23);
            Connect_Spectrum.TabIndex = 10;
            Connect_Spectrum.Text = "connect spectrum";
            Connect_Spectrum.UseVisualStyleBackColor = true;
            Connect_Spectrum.Click += Connect_Spectrum_Click;
            // 
            // SendProgress
            // 
            SendProgress.Location = new Point(37, 333);
            SendProgress.Name = "SendProgress";
            SendProgress.Size = new Size(149, 23);
            SendProgress.TabIndex = 11;
            // 
            // DisconnectPyClient
            // 
            DisconnectPyClient.Location = new Point(205, 230);
            DisconnectPyClient.Name = "DisconnectPyClient";
            DisconnectPyClient.Size = new Size(125, 52);
            DisconnectPyClient.TabIndex = 12;
            DisconnectPyClient.Text = "disconnect python client";
            DisconnectPyClient.UseVisualStyleBackColor = true;
            DisconnectPyClient.Click += DisconnectPyClient_Click;
            // 
            // SpecForm
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(665, 406);
            Controls.Add(DisconnectPyClient);
            Controls.Add(SendProgress);
            Controls.Add(Connect_Spectrum);
            Controls.Add(label4);
            Controls.Add(Message);
            Controls.Add(Publish);
            Controls.Add(Response0);
            Controls.Add(label2);
            Controls.Add(Topic);
            Controls.Add(label1);
            Controls.Add(Broker_addr);
            Controls.Add(Connect_Brocker);
            Name = "SpecForm";
            Text = "Form1";
            Load += MainForm_Load;
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button Connect_Brocker;
        private TextBox Broker_addr;
        private Label label1;
        private TextBox Topic;
        private Label label2;
        private TextBox Response0;
        private Button Publish;
        private TextBox Message;
        private Label label4;
        private Button Connect_Spectrum;
        private ProgressBar SendProgress;
        private Button DisconnectPyClient;
    }
}
