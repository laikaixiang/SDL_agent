using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Timers;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using WinFormsApp_Draft.DK;
using System.IO.Ports;
using System.Reflection.Metadata.Ecma335;
using System.Drawing.Design;
using DocumentFormat.OpenXml.Drawing.Charts;

namespace WinFormsApp_Draft
{
    public partial class DispenserForm : Form
    {
        private SerialPort port = new SerialPort();
        private Axes axes_movement = new Axes();
        private Pipette pip_action = new Pipette();
        private DKPoint point = new DKPoint();
        //private CancellationTokenSource cancellationToken_state;
        public bool port_opened = false;

        private static byte index;
        private const uint baudrate = 115200;

        private const byte left_tip = 1, right_tip = 2;
        private const byte left_z = 3, right_z = 4;
        private const byte x_id = 5, y_id = 6;
        //private static byte left_tip_status = 1, right_tip_status = 1;//1为达到预定位置
        //private static byte left_z_status = 1, right_z_status = 1;
        //private static byte x_status = 1, y_status = 1;
        //private static byte left_pump_status = 1, right_pump_status = 1;

        private byte left_tip_check = 2, right_tip_check = 2;


        public DispenserForm()
        {
            InitializeComponent();
        }

        public async void DispenserForm_Load(object sender, EventArgs e)
        {
            if (port.IsOpen)
            {
                await Task.Run(new Action(() =>
                {
                    Axes.Enable_motor_c(index, x_id, 0);
                    Axes.Enable_motor_c(index, y_id, 0);
                    Axes.Enable_motor_c(index, left_z, 0);
                    Axes.Enable_motor_c(index, right_z, 0);
                    Axes.Enable_motor_c(index, left_tip, 0);
                    Axes.Enable_motor_c(index, right_tip, 0);
                }));
            }

            int response = Axes.CloseComPort(index);
            Response.Text = Convert.ToString(response);
            if (response == 1)
            {
                port_opened = false;
                DispenserSerialSwitch.Text = "connect";
                DispenserConnectionState.Text = "dispenser disconnected";
            }
            DispenserPorts.Items.Clear();

            LeftTipEnable.Enabled = false;
            RightTipEnable.Enabled = false;

            string[] ports = SerialPort.GetPortNames();
            foreach (string port in ports)
            {
                foreach (Match match in Regex.Matches(port, @"\d+"))
                {
                    DispenserPorts.Items.Add(match.Value);
                }
            }
            DispenserPorts.SelectedIndex = 0;

            X.Text = "0";
            Y.Text = "0";
            LeftZ.Text = "0";
            RightZ.Text = "0";
            LeftTipVolume.Text = "0";
            RightTipVolume.Text = "0";
        }

        private async void DispenserSerialSwitch_Click(object sender, EventArgs e)
        {
            string selected_port = DispenserPorts.SelectedItem.ToString();
            index = Convert.ToByte(selected_port);

            if (port_opened)
            {
                await Task.Run(new Action(() =>
                {
                    DispenserConnectionState.Invoke(new Action(() =>
                    {
                        DispenserConnectionState.Text = "disconnecting...";
                    }));
                    Axes.Enable_motor_c(index, x_id, 0);
                    Axes.Enable_motor_c(index, y_id, 0);
                    Axes.Enable_motor_c(index, left_z, 0);
                    Axes.Enable_motor_c(index, right_z, 0);
                    Axes.Enable_motor_c(index, left_tip, 0);
                    Axes.Enable_motor_c(index, right_tip, 0);
                }));

                int response = Axes.CloseComPort(index);
                Response.Text = Convert.ToString(response);
                if (response == 1)
                {
                    port_opened = false;
                    DispenserSerialSwitch.Text = "connect";
                    DispenserConnectionState.Text = "dispenser disconnected";
                }
            }
            else
            {
                int response = Axes.OpenComPort(index, baudrate);
                Response.Text = Convert.ToString(response);

                if (response == 1)
                {
                    port_opened = true;

                    await Task.Run(new Action(() =>
                    {
                        DispenserConnectionState.Invoke(new Action(() =>
                        {
                            DispenserConnectionState.Text = "connecting...";
                        }));
                        Axes.Enable_motor_c(index, x_id, 1);
                        Axes.Enable_motor_c(index, y_id, 1);
                        Axes.Enable_motor_c(index, left_z, 1);
                        Axes.Enable_motor_c(index, right_z, 1);
                        Axes.Enable_motor_c(index, left_tip, 1);
                        Axes.Enable_motor_c(index, right_tip, 1);
                        Axes.Set_speed_ac_de_time_c(index, x_id, 300, 200, 200);
                        Axes.Set_speed_ac_de_time_c(index, y_id, 300, 200, 200);
                        Axes.Set_speed_ac_de_time_c(index, left_z, 800, 200, 200);
                        Axes.Set_speed_ac_de_time_c(index, right_z, 800, 200, 200);

                        Pipette.Cpump_zero_p(index, left_tip);
                        Pipette.Cpump_zero_p(index, right_tip);
                    }));

                    check_pipette();
                    await reset_dispenser();
                    DispenserSerialSwitch.Text = "disconnect";
                    DispenserConnectionState.Text = "dispenser connected";

                }
                else
                {
                    Response.Text = "connection failed";
                }
            }
        }

        private async void ResetDispensor_Click(object sender, EventArgs e)
        {
            check_pipette();
            await reset_dispenser();
            Response.Text = "reset done";
        }

        //reset the dispenser, moving it back to zero point and dropping the tips, but not spitting liquid even if residual
        public async Task reset_dispenser()
        {
            LeftTipEnable.Enabled = false;
            RightTipEnable.Enabled = false;

            Axes.Zero_c(index, left_z);
            Axes.Zero_c(index, right_z);
            await Task.Delay(4000);
            Axes.Zero_c(index, x_id);
            Axes.Zero_c(index, y_id);

            back_tip();
        }

        private async void AxesManeuver_front_Click(object sender, EventArgs e)
        {
            point.x = Convert.ToInt32(X.Text);
            point.y = Convert.ToInt32(Y.Text);
            point.lz = Convert.ToInt32(LeftZ.Text);
            point.rz = Convert.ToInt32(RightZ.Text);
            await Task.Run(() => MovL_hor(point));
            await Task.Run(() => MovL_ver(point));
            check_pipette();
        }
        private async void AxesManeuver_back_Click(object sender, EventArgs e)
        {
            point.x = Convert.ToInt32(X.Text);
            point.y = Convert.ToInt32(Y.Text);
            point.lz = Convert.ToInt32(LeftZ.Text);
            point.rz = Convert.ToInt32(RightZ.Text);
            await Task.Run(() => MovL_ver(point, 3000));
            await Task.Run(() => MovL_hor(point));
            check_pipette();
        }

        //entrance for auto moving in back and forth 
        /// <summary>
        /// write the point name down and the dispenser will move to the correlated point, going horizontally
        /// points are defined in DispenserPoints.json
        /// </summary>
        /// <param name="point"></param>
        /// <returns></returns>
        public async Task MovL_hor(DKPoint point)
        {
            if (point == null) return;
            else
            {
                Axes.Motor_Absolute_movement_c(index, x_id, point.x);
                Axes.Motor_Absolute_movement_c(index, y_id, point.y);
                await Task.Delay(2000);

                //Axes.Motor_Absolute_movement_c(index, left_z, point.lz);
                //Axes.Motor_Absolute_movement_c(index, right_z, point.rz);
                //await Task.Delay(1500);
            }
        }
        /// <summary>
        /// <para>
        /// write the point name down and the dispenser will move to the correlated point, going vertically
        /// points are defined in DispenserPoints.json. which means moving left pipette or right depends.
        /// </para>
        /// <para>
        /// if delay is passed in and not 0, will wait until finishing moving the actual time(ms); 
        /// if kept default, will calculate the wait time according to how much to go down along side the "z" axis.
        /// the waiting time will be returned, but will not move
        /// </para>
        /// <para>
        /// if pipette is 1, left pipette will do the operation. if 2, the right. right by default
        /// </para>
        /// </summary>
        /// <param name="point"></param>
        /// <returns></returns>
        public async Task<int> MovL_ver(DKPoint point, int delay = 0, byte pipette = right_tip)
        {
            if (point == null) return 0;
            else
            {
                if (delay == 0)
                {
                    if (pipette == left_tip)
                    {
                        int real_delay = point.lz / 100;
                        return real_delay;
                    }
                    else if (pipette == right_tip)
                    {
                        int real_delay = point.rz / 100;
                        return real_delay;
                    }
                    else
                    {
                        return 0;
                    }
                }
                else
                {
                    Axes.Motor_Absolute_movement_c(index, left_z, point.lz);
                    Axes.Motor_Absolute_movement_c(index, right_z, point.rz);
                    await Task.Delay(delay);
                    return delay;
                }
            }
        }

        private void CheckTip_Click(object sender, EventArgs e)
        {
            check_pipette();
        }

        //check if tip is on specific pipette
        /// <summary>
        /// 0: no tip available; 1: left available; 2: right available; 3: both available 
        /// </summary>
        /// <returns></returns>
        public int check_pipette()
        {
            Pipette.Find_tip_p(index, left_tip, ref left_tip_check);
            Pipette.Find_tip_p(index, right_tip, ref right_tip_check);
            string response = "";
            int flag = 0;

            if (left_tip_check != 2)
            {
                LeftTipEnable.Enabled = true;
                response += "Left pipette available.";
                flag += 1;
            }
            else
            {
                LeftTipEnable.Enabled = false;
            }

            if (right_tip_check != 2)
            {
                RightTipEnable.Enabled = true;
                response += "Right pipette available.";
                flag += 2;
            }
            else
            {
                RightTipEnable.Enabled = false;
            }

            if (flag == 0)
            {
                response = "No pipettes available";
            }
            Response.Text = response;
            return flag;
        }

        //auto spit and suck
        /// <summary>
        /// the unit of volume is ul; 1 for left tip and 2 for right;
        /// max of volume is 32767, otherwise higher bits will be cut off
        /// </summary>
        /// <param name="volume"></param>
        /// <param name="tip"></param>
        /// <returns></returns>
        public async Task Tip_Suck(int volume, byte tip = right_tip)
        {
            byte return_value = 0;
            //int flag = check_pipette();
            //if (flag == tip || flag == 3)
            //{
            if (tip == left_tip)
            {
                Int16 vol_int16 = Convert.ToInt16(volume);
                Pipette.Suck_p(index, tip, vol_int16, ref return_value);
                await Task.Delay(1000);
            }
            else if (tip == right_tip)
            {
                Int16 vol_int16 = Convert.ToInt16(volume);
                Pipette.Suck_p(index, tip, vol_int16, ref return_value);
                await Task.Delay(1000);
            }
            //}
        }

        public async Task Tip_Spit(int volume, byte tip = right_tip)
        {
            byte return_value = 0;
            //int flag = check_pipette();
            //if (flag == tip || flag == 3)
            //{
            if (tip == left_tip)
            {
                Int16 vol_int16 = Convert.ToInt16(volume);
                Pipette.Spit_p(index, tip, vol_int16, ref return_value);
                await Task.Delay(500);
            }
            else if (tip == right_tip)
            {
                Int16 vol_int16 = Convert.ToInt16(volume);
                Pipette.Spit_p(index, tip, vol_int16, ref return_value);
                await Task.Delay(500);
            }
            //}
        }

        public void back_tip(byte tip = right_tip)
        {
            if(tip == left_tip)
            {
                Pipette.Back_tip_p(index, left_tip);
                left_tip_check = 2;
            }
            else if(tip == right_tip)
            {
                Pipette.Back_tip_p(index, right_tip);
                right_tip_check = 2;
            }
            else
            {
                Pipette.Back_tip_p(index, left_tip);
                left_tip_check = 2;
                Pipette.Back_tip_p(index, right_tip);
                right_tip_check = 2;
            }
        }

        private void LeftTipSuck_Click(object sender, EventArgs e)
        {
            byte return_value = 0;
            Int16 volume = Convert.ToInt16(LeftTipVolume.Text);
            Pipette.Suck_p(index, left_tip, volume, ref return_value);
        }

        private void LeftTipSpit_Click(object sender, EventArgs e)
        {
            byte return_value = 0;
            Int16 volume = Convert.ToInt16(LeftTipVolume.Text);
            Pipette.Spit_p(index, left_tip, volume, ref return_value);
        }

        private void LeftTipDrop_Click(object sender, EventArgs e)
        {
            Pipette.Back_tip_p(index, left_tip);
            left_tip_check = 2;
            LeftTipEnable.Enabled = false;
        }

        private void RightTipSuck_Click(object sender, EventArgs e)
        {
            byte return_value = 0;
            Int16 volume = Convert.ToInt16(RightTipVolume.Text);
            Pipette.Suck_p(index, right_tip, volume, ref return_value);
        }

        private void RightTipSpit_Click(object sender, EventArgs e)
        {
            byte return_value = 0;
            Int16 volume = Convert.ToInt16(RightTipVolume.Text);
            Pipette.Spit_p(index, right_tip, volume, ref return_value);
        }

        private void RightTipDrop_Click(object sender, EventArgs e)
        {
            Pipette.Back_tip_p(index, right_tip);
            right_tip_check = 2;
            RightTipEnable.Enabled = false;
        }
    }
}
