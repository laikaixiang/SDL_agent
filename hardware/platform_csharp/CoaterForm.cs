using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Modbus.Device;
using System.IO.Ports;
using System.Timers;
using DocumentFormat.OpenXml.Presentation;

namespace WinFormsApp_Draft
{
    public partial class CoaterForm : Form
    {
        //coater declarations
        private static SerialPort motor_port = new SerialPort();
        public IModbusMaster master;

        private static int static_spin_speed = 0;
        private static int static_acc_speed = 0;
        private static int static_spin_dur = 0;
        private System.Timers.Timer spin_timer = new System.Timers.Timer(); 
        private CancellationTokenSource cancellationTokenSource_pos;
        private CancellationTokenSource cancellationTokenSource_beat;
        public bool coater_connect_state = false;

        public CoaterForm()
        {
            InitializeComponent();
        }

        public void CoaterForm_Load(object sender, EventArgs e) 
        {
            if (motor_port.IsOpen)
            {
                motor_port.Close();
                MotorSerialSwitch.Text = "Open Serial";
            }
            MotorBaudRate.Items.Clear();
            MotorPorts.Items.Clear();
            DisableCoater();

            string[] ports = SerialPort.GetPortNames();
            foreach (string port in ports)
            {
                MotorPorts.Items.Add(port);
            }
            MotorPorts.SelectedIndex = 0;

            MotorBaudRate.Items.Add("9600");
            MotorBaudRate.Items.Add("19200");
            MotorBaudRate.Items.Add("38400");
            MotorBaudRate.Items.Add("115200");
            MotorBaudRate.SelectedIndex = 0; 
        }

        public void EnableCoater()
        {
            foreach (System.Windows.Forms.Control ctr in Controls)
            {
                if (ctr == SpinCoater)
                {
                    ctr.Enabled = true;
                }
            }
        }

        public void DisableCoater()
        {
            foreach (System.Windows.Forms.Control ctr in Controls)
            {
                if (ctr == SpinCoater)
                {
                    ctr.Enabled = false;
                }
            }
        }

        private void MotorPorts_SelectedIndexChanged(object sender, EventArgs e)
        {
            SelectedMotorPort.Text = MotorPorts.SelectedItem.ToString();
        }

        private void MotorBaudRate_SelectedIndexChanged(object sender, EventArgs e)
        {
            SelectedMotorBR.Text = MotorBaudRate.SelectedItem.ToString();
        }

        //connect the motor
        private void MotorSerialSwitch_Click(object sender, EventArgs e)
        {
            if (SelectedMotorBR.Text != "" && SelectedMotorPort.Text != "" && motor_port.IsOpen == false)
            {
                motor_port.PortName = SelectedMotorPort.Text;
                motor_port.BaudRate = Convert.ToInt32(SelectedMotorBR.Text);
                motor_port.DataBits = 8;
                motor_port.StopBits = StopBits.Two;
                motor_port.Parity = Parity.None;
                motor_port.WriteTimeout = 500;
                motor_port.ReadTimeout = 500;

                try
                {
                    Response.Clear();
                    motor_port.Open();
                    master = ModbusSerialMaster.CreateRtu(motor_port);
                    MotorConnectionState.Text = "connecting..";

                    if (motor_port.IsOpen)
                    {   
                        EnableCoater();
                        cancellationTokenSource_pos = new CancellationTokenSource();
                        cancellationTokenSource_beat = new CancellationTokenSource();   

                        MotorSerialSwitch.Text = "Disconnect Motor";
                        MotorConnectionState.Text = "opened";
                        coater_connect_state = true;
                        SpinSpeed.Text = "2000";
                        SpinDuration.Text = "0";
                        AccSpeed.Text = "1000";

                        Task.Run(() => StartBeatAsync(master, cancellationTokenSource_beat.Token));

                        byte slaveID = 0x01;
                        ushort modeAddress = 0x1771;
                        master.WriteSingleRegister(slaveID, modeAddress, 0x0006);
                    }
                }
                catch (Exception ex)
                {
                    Response.Text = ex.Message;
                    motor_port.Close();
                    MotorSerialSwitch.Text = "Open Serial";
                    MotorConnectionState.Text = "closed";
                    DisableCoater();
                }
        }
            else
            {
                try
                {
                    Response.Clear();
                    MotorSerialSwitch.Text = "Open Serial";
                    MotorConnectionState.Text = "closed";
                    if (cancellationTokenSource_beat != null)
                    {
                        cancellationTokenSource_beat.Cancel();
                    }
                    if(cancellationTokenSource_pos != null)
                    {
                        cancellationTokenSource_pos.Cancel();
                    }
                    coater_connect_state = false;
                    motor_port.Close();
                    DisableCoater();
                }
                catch (Exception ex)
                {
                    Response.Text = ex.Message;
                }
            }
        }

        //start driver heart beat 
        private static async Task StartBeatAsync(IModbusMaster master, CancellationToken cancellationToken)
        {
            byte slaveID = 0x01;
            ushort beatAddress = 0x1770;
            await Task.Run(async () =>
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    master.WriteSingleRegister(slaveID, beatAddress, 0x0001);
                    await Task.Delay(100);
                    master.WriteSingleRegister(slaveID, beatAddress, 0x0002);
                    await Task.Delay(100);
                }
            });
        }

        //start the motor
        private async Task Send_Request()
        {
            if (motor_port.IsOpen == true)
            {
                try
                {
                    byte slaveID = 0x01;

                    if (static_spin_speed != 0)
                    {
                        ushort speedAddress = 0x1773;
                        int e_speed = static_spin_speed * 7;
                        ushort high = (ushort)(e_speed >> 16);
                        ushort low = (ushort)e_speed;
                        ushort[] speed_erpm = { high, low };

                        master.WriteMultipleRegisters(slaveID, speedAddress, speed_erpm);
                    }

                    if (static_acc_speed != 0)
                    {
                        ushort accAddress = 0x1780;
                        int e_acc = static_acc_speed * 7;
                        ushort high = (ushort)(e_acc >> 16);
                        ushort low = (ushort)e_acc;
                        ushort[] acc_erpm = { high, low };
        
                        master.WriteMultipleRegisters(slaveID, accAddress, acc_erpm);
                    }

                    else
                    {
                        Response.Invoke(() =>
                        {
                            Response.Text += Convert.ToString(static_acc_speed);
                        });
                    }

                    if (static_spin_dur != 0)
                    {
                        spin_timer.Elapsed += new ElapsedEventHandler(FreeStop_timer);
                        spin_timer.Interval = static_spin_dur * 1000;
                        spin_timer.AutoReset = false;
                        spin_timer.Start();
                    }

                    ushort modeAddress = 0x1771;
                    master.WriteSingleRegister(slaveID, modeAddress, 0x0001);
                }

                catch (Exception ex)
                {
                    if (Response.InvokeRequired)
                    {
                        Response.Invoke(new Action(() =>
                        {
                            Response.Text = ex.Message;
                        }));
                    }
                    else
                    {
                        Response.Text = ex.Message;
                    }
                }
            }
        }

        //entrance for auto spin coating
        /// <summary>
        /// spin_speed rpm; acc_speed rpm/s; spin_dur s
        /// </summary>
        /// <param name="spin_speed"></param>
        /// <param name="acc_speed"></param>
        /// <param name="spin_dur"></param>
        /// <returns></returns>
        public async Task Spin_Coat(int spin_speed, int acc_speed, int spin_dur)
        {
            static_spin_speed = spin_speed;
            static_acc_speed = acc_speed;
            static_spin_dur = spin_dur;
            await Task.Run(() => Send_Request());
            await Task.Delay(spin_dur);
            await Task.Run(() => ResetMotorAsync(master, spin_speed));
        }

        private async void SendRequest_Click(object sender, EventArgs e)
        {
            static_spin_speed = Convert.ToInt32(SpinSpeed.Text);
            static_acc_speed = Convert.ToInt32(AccSpeed.Text);
            static_spin_dur = Convert.ToInt32(SpinDuration.Text);
            await Task.Run(() => Send_Request());
        }

        private void FreeStop_timer(object? sender, ElapsedEventArgs e)
        {
            byte slaveID = 0x01;
            ushort stopAddress = 0x177E;
            ushort modeAddress = 0x1771;
            master.WriteSingleRegister(slaveID, stopAddress, 0x0000);
            master.WriteSingleRegister(slaveID, modeAddress, 0x0006);
            spin_timer.Stop();

            static_spin_speed = 0;
        }

        //release the motor if button pressed
        private async void FreeStop_Click(object sender, EventArgs e)
        {
            await Task.Run(() => FreeStopAsync(FreeStop, master));
            static_spin_speed = 0;
        }

        private async Task FreeStopAsync(Button freestop, IModbusMaster master)
        {
            byte slaveID = 0x01;
            ushort stopAddress = 0x177E;
            ushort modeAddress = 0x1771;
            master.WriteSingleRegister(slaveID, stopAddress, 0x0000);
            master.WriteSingleRegister(slaveID, modeAddress, 0x0006);
        }

        //stop the motor with brutal force if button pressed
        private void ForceStop_Click(object sender, EventArgs e)
        {
            Task.Run(() => ForceStopAsync(ForceStop, master));
            static_spin_speed = 0;
        }

        private async Task ForceStopAsync(Button forcestop, IModbusMaster master)
        {
            byte slaveID = 0x01;
            ushort stopAddress = 0x177E;
            ushort modeAddress = 0x1771;
            master.WriteSingleRegister(slaveID, stopAddress, 0x0064);
            master.WriteSingleRegister(slaveID, modeAddress, 0x0006);
        }

        //show motor position at live if checkbox checked
        private async void Update_CheckedChanged(object sender, EventArgs e)
        {
            if (Update.Checked)
            {
                await Task.Run(() => UpdatePositionAsync(Position, master, cancellationTokenSource_pos.Token));
            }
            else
            {
                cancellationTokenSource_pos.Cancel();
                Position.Clear();
                cancellationTokenSource_pos = new CancellationTokenSource();
            }
        }

        private async Task UpdatePositionAsync(TextBox textBox, IModbusMaster master, CancellationToken cancellationToken)
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    byte slaveID = 0x01;
                    ushort posAddress = 0x1392;
                    ushort[] pos = master.ReadInputRegisters(slaveID, posAddress, 2);
                    int high = pos[0];
                    int low = pos[1];
                    int cur_pos = ((high << 16) + low) / 100 + 1;
                    
                    textBox.Invoke(() =>
                    {
                       textBox.Text = Convert.ToString(cur_pos);
                    });
                    await Task.Delay(100);
                }
                catch { }
            }
        }

        //set current position as motor's zero point if button pressed
        private async void ClearPos_Click(object sender, EventArgs e)
        {
            Position.Clear();
            await Task.Run(() => ClearPosAsync(master));
        }

        private async Task ClearPosAsync(IModbusMaster master)
        {
            byte slaveID = 0x01;
            ushort registerAddress_write = 0x177c;
            ushort[] val = { 0x0000, 0x0000 };
            master.WriteMultipleRegisters(slaveID, registerAddress_write, val);
        }

        private async void ResetMotor_Click(object sender, EventArgs e)
        {
            await Task.Run(() => ResetMotorAsync(master, static_spin_speed));
        }

        //reset motor to zero
        private async Task ResetMotorAsync(IModbusMaster master, int spin_speed = 0)
        {
            try
            {
                await Task.Run(() => slow_down(master));
                await Task.Delay(spin_speed * 2 + 500);
                await Task.Run(() => reset_to_pos(master));
                await Task.Delay(500);
            }
            catch { }
        }

        private async Task slow_down(IModbusMaster maseter)
        {
            byte slaveID = 0x01;
            ushort modeAddress = 0x1771;
            ushort speedAddress = 0x1773;

            int e_speed = 0;
            ushort high0 = (ushort)(e_speed >> 16);
            ushort low0 = (ushort)e_speed;
            ushort[] speed_erpm = { high0, low0 };

            master.WriteMultipleRegisters(slaveID, speedAddress, speed_erpm);
            master.WriteSingleRegister(slaveID, modeAddress, 0x0001);
        }

        private async Task reset_to_pos(IModbusMaster master)
        {
            byte slaveID = 0x01;
            ushort modeAddress = 0x1771;
            ushort cur_posAddress = 0x1392;
            ushort resetAddress = 0x1776;

            ushort[] pos = master.ReadInputRegisters(slaveID, cur_posAddress, 2);
            int high = pos[0];
            int low = pos[1];
            int cur_pos = ((high << 16) + low) / 100 + 1;

            int reset_pos = 100 * (cur_pos - (cur_pos % 360) + 360);
            ushort high_convert = (ushort)(reset_pos >> 16);
            ushort low_convert = (ushort)reset_pos;
            ushort[] reset_to = { high_convert, low_convert };

            master.WriteMultipleRegisters(slaveID, resetAddress, reset_to);
            master.WriteSingleRegister(slaveID, modeAddress, 0x0003);

            static_spin_speed = 0;
        }
    }
}
