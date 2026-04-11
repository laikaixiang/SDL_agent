using System.IO;
using System.IO.Ports;
using System.Net.Sockets;
using System.Timers;
using Modbus.Device;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using CSharpTcpDemo;
using CSharpTcpDemo.com.dobot.api;
using CSharthiscpDemo.com.dobot.api;
using WinFormsApp_Draft.Auto;
using WinFormsApp_Draft.DK;
using Winform_platform.Auto;
using uPLibrary.Networking.M2Mqtt;
using Microsoft.VisualBasic;
using System.Threading.Tasks;


namespace WinFormsApp_Draft
{
    public partial class MainForm : Form
    {
        //general declarations
        private SerialPort motor_port = new SerialPort();
        private IModbusMaster master;

        //coater declarations
        private static int spin_speed = 0;
        private System.Timers.Timer spin_timer = new System.Timers.Timer();
        private CancellationTokenSource cancellationTokenSource_pos;
        private CancellationTokenSource cancellationTokenSource_beat;

        //subforms
        private ArmForm armForm = new ArmForm();
        private CoaterForm coaterForm = new CoaterForm();
        private DispenserForm dispenserForm = new DispenserForm();
        private SpecForm specForm = new SpecForm();

        //json configurations
        private class ArmConfig
        {
            public Dictionary<string, DescartesPoint> Points { get; set; } = new Dictionary<string, DescartesPoint>();
        }

        private class DispenserConfig
        {
            public Dictionary<string, DKPoint> Points { get; set; } = new Dictionary<string, DKPoint>();
        }

        private const string armPath = "ArmPoints.json";
        private static string arm_json = File.ReadAllText(armPath);
        private ArmConfig? arm_conf = JsonConvert.DeserializeObject<ArmConfig>(arm_json);

        private const string dispenserPath = "DispenserPoints.json";
        private static string dispenser_json = File.ReadAllText(dispenserPath);
        private DispenserConfig? dispenser_conf = JsonConvert.DeserializeObject<DispenserConfig>(dispenser_json);

        private const string reagentPath = "reagent_layout.json";

        //auto declarations
        private static MqttClient client = SpecForm.client;

        private static List<List<int>> exp_rounds = new List<List<int>>();//list of coater parameters for each round
        private static List<List<int>> features = new List<List<int>>();//list of volume of reagents and antisolvents
        private static List<string> free_substrates = new List<string>();
        private static List<string> free_right_tips = new List<string>();

        private static Dictionary<string, List<int>> reagent = new Dictionary<string, List<int>>();//dict of reagent pos to reagent volume for right pump
        private static Dictionary<string, List<int>> dispense_time = new Dictionary<string, List<int>>();//dict of reagent pos to when to spit
        private static int feature_count = 3;
        private static int running = 0;

        public MainForm()
        {
            InitializeComponent();

            Refresh.Click += armForm.ArmForm_Load;
            Refresh.Click += dispenserForm.DispenserForm_Load;
            Refresh.Click += coaterForm.CoaterForm_Load;
            Refresh.Click += MainForm_Load;
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            exp_rounds.Clear();
            free_substrates.Clear();
            free_right_tips.Clear();
            reagent.Clear();
            ReagentLayout.Controls.Clear();
            ExperimentParameters.Items.Clear();
            Log.Clear();
            Method.Controls.Clear();
            AutoRead.Checked = false;
            Method.Enabled = false;
            ReagentFeatures.Enabled = false;

            reload_substrate();
            reload_tipbox();

            init_btns(6, 4);
            if (!Method.Contains(ReagentLayout)) Method.Controls.Add(ReagentLayout);
            if (!Method.Contains(ExperimentParameters)) Method.Controls.Add(ExperimentParameters);
            if (!Method.Contains(ReagentFeatures)) Method.Controls.Add(ReagentFeatures);
            if (!Method.Contains(Log)) Method.Controls.Add(Log);
            if (!Method.Contains(NameReagent)) Method.Controls.Add(NameReagent);
            if (!Method.Contains(ClearReagent)) Method.Controls.Add(ClearReagent);
            if (!Method.Contains(label3)) Method.Controls.Add(label3);
            if (!Method.Contains(label4)) Method.Controls.Add(label4);
            if (!Method.Contains(label5)) Method.Controls.Add(label5);
            if (!Method.Contains(label6)) Method.Controls.Add(label6);
            Log.Enabled = false;
            NameReagent.Enabled = false;

            string strPath = System.Windows.Forms.Application.StartupPath + "\\";
            ErrorInfoHelper.ParseControllerJsonFile(strPath + "alarm_controller.json");
            ErrorInfoHelper.ParseServoJsonFile(strPath + "alarm_servo.json");

            FilePath.Text = "C:\\Users\\DELL\\Desktop\\readtest.csv";
            SheetName.Text = "Sheet1";

            armForm.TopLevel = false;
            coaterForm.TopLevel = false;
            dispenserForm.TopLevel = false;
            specForm.TopLevel = false;
        }

        private void reload_tipbox()
        {
            free_right_tips.Clear();

            foreach (string pos in dispenser_conf.Points.Keys)
            {
                if (pos.StartsWith('R'))
                {
                    free_right_tips.Add(pos);
                }

            }
        }

        private void reload_substrate()
        {
            free_substrates.Clear();
            foreach (string pos in arm_conf.Points.Keys)
            {
                if (pos.StartsWith('P'))
                {
                    free_substrates.Add(pos);
                }
            }
        }

        private void Refresh_Click(object sender, EventArgs e)
        {

        }

        private void Tipbox_Reload_Click(object sender, EventArgs e)
        {
            reload_tipbox();
        }

        private void Substrate_Reload_Click(object sender, EventArgs e)
        {
            reload_substrate();
        }

        //layout of the reagent bottles
        private void init_btns(int row_n, int col_n)
        {
            ReagentLayout.ColumnCount = col_n;
            ReagentLayout.RowCount = row_n;
            set_size(row_n, col_n);

            Button[,] buttons = new Button[row_n, col_n];

            for (int i = 0; i < row_n; i++)
            {
                for (int j = 0; j < col_n; j++)
                {
                    buttons[i, j] = new Button();
                    buttons[i, j].Text = $"BP{i + 1}{j + 1}";
                    buttons[i, j].Dock = DockStyle.Fill;
                    buttons[i, j].BackColor = Color.Red;

                    ReagentLayout.Controls.Add(buttons[i, j], j, i);
                    buttons[i, j].MouseClick += (sender, e) =>
                    {
                        Button button = (Button)sender;
                        add_liquid(button);
                    };
                }
            }
        }

        private void set_size(int row_n, int col_n)
        {
            ReagentLayout.ColumnStyles.Clear();
            ReagentLayout.RowStyles.Clear();
            for (int i = 0; i < col_n; i++)
            {
                ReagentLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F / col_n));
            }
            for (int j = 0; j < row_n; j++)
            {
                ReagentLayout.RowStyles.Add(new ColumnStyle(SizeType.Percent, 100F / row_n));
            }
        }

        private async void add_liquid(Button button)
        {
            try
            {
                if (button.BackColor == Color.Red)
                {
                    button.BackColor = Color.Yellow;
                    ReagentFeatures.Enabled = true;
                    ReagentFeatures.SelectedIndexChanged += selecting_reagent_vol;
                    int index = 0;

                    NameReagent.Enabled = true;

                    while (true)
                    {
                        if (ReagentFeatures.BackColor == Color.Yellow)
                        {
                            string reagent_name = NameReagent.Text;
                            NameReagent.Text = "";
                            NameReagent.Enabled = false;
                            if (reagent_name.Length != 0)
                            {
                                Platform_Config.record_reagent(reagentPath, button.Text, reagent_name);
                                Log.Text += System.String.Format("reagent at {0} is {1}", button.Text, reagent_name) + "\n";
                            }

                            button.Enabled = false;
                            index = ReagentFeatures.SelectedIndex;
                            ReagentFeatures.SelectedIndexChanged -= selecting_reagent_vol;
                            ReagentFeatures.BackColor = Color.White;

                            reagent.Add(button.Text, features[index]);

                            //button.BackColor = Color.ForestGreen;
                            //Log.Text += System.String.Format("reagent {0}: {1}", button.Text, ReagentFeatures.SelectedItem) + "\n";
                            //Response.Text = button.Text + System.String.Format(" added as reagent, correlated with feature index {0}", index);

                            //set checkpoint dict
                            button.BackColor = Color.Blue;
                            ReagentFeatures.SelectedIndexChanged += selecting_reagent_time;

                            while (true)
                            {
                                if (ReagentFeatures.BackColor == Color.Blue)
                                {
                                    index = ReagentFeatures.SelectedIndex;
                                    dispense_time.Add(button.Text, features[index]);
                                    ReagentFeatures.SelectedIndexChanged -= selecting_reagent_time;
                                    ReagentFeatures.BackColor = Color.White;

                                    button.BackColor = Color.ForestGreen;
                                    Log.Text += System.String.Format("dispense time of reagent at {0}: {1}", button.Text, ReagentFeatures.SelectedItem) + "\n";
                                    Response.Clear();
                                    break;
                                }
                                if (button.BackColor == Color.Red)
                                {
                                    break;
                                }
                                await Task.Delay(100);
                                Response.Text = button.Text + "choosing dispense time";
                            }
                            button.Enabled = true;
                            break;
                        }
                        else if (button.BackColor == Color.Red)
                        {
                            Response.Text = button.Text + "quit";
                            ReagentFeatures.Enabled = false;
                            if (reagent.ContainsKey(button.Text))
                            {
                                reagent.Remove(button.Text);
                            }
                            break;
                        }
                        Response.Text = button.Text + "choosing reagent volume";
                        await Task.Delay(100);
                    }
                    NameReagent.Text = "";
                    NameReagent.Enabled = false;
                }
                else
                {
                    string reagent_name = NameReagent.Text;
                    NameReagent.Text = "";
                    NameReagent.Enabled = false;
                    if (reagent_name.Length != 0)
                    {
                        Platform_Config.record_reagent(reagentPath, button.Text, reagent_name);
                        Log.Text += System.String.Format("reagent at {0} is {1}", button.Text, reagent_name) + "\n";
                    }
                    else
                    {
                        Platform_Config.record_reagent(reagentPath, button.Text, "");
                        Log.Text += System.String.Format("reagent at {0} is removed", button.Text) + "\n";
                    }

                    button.BackColor = Color.Red;

                    if (reagent.Keys.Contains(button.Text))
                    {
                        reagent.Remove(button.Text);
                        Log.Text += button.Text + " removed" + "\n";
                    }
                    if (dispense_time.Keys.Contains(button.Text))
                    {
                        dispense_time.Remove(button.Text);
                        Log.Text += "dispense time " + button.Text + " removed" + "\n";
                    }
                    ReagentFeatures.Enabled = false;
                }
            }
            catch { }
        }

        private void selecting_reagent_vol(object sender, EventArgs e)
        {
            ReagentFeatures.BackColor = Color.Yellow;
        }

        private void selecting_reagent_time(object sender, EventArgs e)
        {
            ReagentFeatures.BackColor = Color.Blue;
        }


        private void ClearReagent_Click(object sender, EventArgs e)
        {
            Platform_Config.clear_all_reagents(reagentPath);
            Log.Text += "all reagents cleared from layout pannel" + "\n";
        }

        // auto read functions, complete automatic
        private void AutoRead_CheckedChanged(object sender, EventArgs e)
        {
            if (AutoRead.Checked && feature_count != 0)
            {
                try
                {
                    int i = 1, x = 3;
                    while (true)
                    {
                        //test data reading process
                        List<int> row_paramerts = ReadExcel.row_param(i, feature_count, FilePath.Text);
                        if (row_paramerts[0] != -1)
                        {
                            exp_rounds.Add(row_paramerts);
                            i++;
                            string round_params = "";
                            for (int j = 0; j < row_paramerts.Count(); j++)
                            {
                                if (j < row_paramerts.Count() - 1)
                                {
                                    round_params += row_paramerts[j].ToString() + ",";
                                }
                                else
                                {
                                    round_params += row_paramerts[j].ToString();
                                }
                            }
                            ExperimentParameters.Items.Add(round_params);
                        }
                        else
                        {
                            break;
                        }

                        if (exp_rounds.Count > 0)
                        {
                            Method.Enabled = true;
                        }
                        else
                        {
                            Method.Enabled = false;
                        }
                    }
                    while (true)
                    {
                        List<int> col_paramerts = ReadExcel.col_param(x, i - 1, FilePath.Text);
                        if (col_paramerts[0] != -1)
                        {
                            features.Add(col_paramerts);
                            x++;
                            string round_params = "";
                            for (int y = 0; y < col_paramerts.Count(); y++)
                            {
                                if (y < col_paramerts.Count() - 1)
                                {
                                    round_params += col_paramerts[y].ToString() + ",";
                                }
                                else
                                {
                                    round_params += col_paramerts[y].ToString();
                                }
                            }
                            ReagentFeatures.Items.Add(round_params);
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Response.Text = ex.Message;
                }
            }
            else
            {
                if (armForm.arm_enable_state)
                {
                    armForm.EnableWindow();
                }
                if (coaterForm.coater_connect_state)
                {
                    coaterForm.EnableCoater();
                }
                Response.Clear();
                ExperimentParameters.Items.Clear();
                ReagentFeatures.Items.Clear();
                Method.Enabled = false;
            }
        }

        private void OpenArmForm_Click(object sender, EventArgs e)
        {
            Open_Form(armForm);
        }

        private void OpenCoaterForm_Click(object sender, EventArgs e)
        {
            Open_Form(coaterForm);
        }
        private void OpenDispenserForm_Click(object sender, EventArgs e)
        {
            Open_Form(dispenserForm);
        }
        private void OpenSpecForm_Click(object sender, EventArgs e)
        {
            Open_Form(specForm);
        }


        private void Open_Form(Form form)
        {
            if (ControlPanel.Controls.Contains(form))
            {
                form.BringToFront();
            }
            else
            {
                form.FormBorderStyle = FormBorderStyle.None;
                SetWindowSize(form);
                ControlPanel.Controls.Add(form);
                form.Show();
                form.BringToFront();
            }
        }

        private void SetWindowSize(Form form)
        {
            double x = this.ControlPanel.Width / Convert.ToSingle(form.Width);
            double y = this.ControlPanel.Height / Convert.ToSingle(form.Height);
            if (form.Controls.Count > 0)
            {
                foreach (System.Windows.Forms.Control control in form.Controls)
                {
                    if (control.Controls.Count > 0)
                    {
                        foreach (System.Windows.Forms.Control c in control.Controls)
                        {
                            c.Width = Convert.ToInt32(c.Width * x);
                            c.Height = Convert.ToInt32(c.Height * y);
                            c.Left = Convert.ToInt32(c.Left * x);
                            c.Top = Convert.ToInt32(c.Top * y);
                        }
                    }
                    control.Width = Convert.ToInt32(control.Width * x);
                    control.Height = Convert.ToInt32(control.Height * y);
                    control.Left = Convert.ToInt32(control.Left * x);
                    control.Top = Convert.ToInt32(control.Top * y);
                }
            }
            form.Width = Convert.ToInt32(form.Width * x);
            form.Height = Convert.ToInt32(form.Height * y);
        }

        //test arm and dispenser, do group move round
        /// <summary>
        /// <para>
        /// pass in point name(string);
        /// </para>
        /// <para>
        /// if gripper is "pick at tray", arm will descend to substrate tray and grip, if "release at tray", vice versa.
        /// if "pick at coater" or "release at coater", similar to substrates, but will descend to the height of the coater.
        /// otherwise, gripper will do nothing, only arm will move.
        /// </para>
        /// <para>
        /// "pick at tray" removes picked substrate from free_substrates; 
        /// "release at tray" adds where the substrate is put back to free_substrates
        /// </para>
        /// </summary>
        /// <param name="point"></param>
        /// <param name="gripper"></param>
        /// <returns></returns>
        private async Task arm_MovL(string point, string gripper = "none")
        {
            if (free_substrates.Count == 0)
            {
                return;
            }
            DescartesPoint arm_pt = new DescartesPoint();
            arm_conf.Points.TryGetValue(point, out arm_pt);
            await armForm.MovL(arm_pt);

            if (gripper == "pick at tray" && free_substrates.Contains(point))
            {
                arm_pt.z = 66;
                arm_pt.r = 5;
                await armForm.MovL(arm_pt);
                await armForm.Grip(13);
                arm_pt.z = 200;
                await armForm.MovL(arm_pt);
                free_substrates.Remove(point);
            }
            else if (gripper == "release at tray" && !free_substrates.Contains(point))
            {
                arm_pt.z = 65;
                arm_pt.r = 5;
                await armForm.MovL(arm_pt);
                await armForm.Release(13);
                arm_pt.z = 200;
                await armForm.MovL(arm_pt);
                free_substrates.Add(point);
            }
            else if (gripper == "pick at coater")
            {
                arm_pt.z = 116;
                arm_pt.r = 8;
                await armForm.MovL(arm_pt);
                await armForm.Grip(13);
                arm_pt.z = 200;
                await armForm.MovL(arm_pt);
            }
            else if (gripper == "release at coater")
            {
                arm_pt.z = 116;
                arm_pt.r = 8;
                await armForm.MovL(arm_pt);
                await armForm.Release(13);
                arm_pt.z = 200;
                await armForm.MovL(arm_pt);
            }
        }

        /// <summary>
        /// <para>
        /// pass in point name(string);
        /// </para>
        /// <para>
        /// if tip is "get", will descend to the pipette box and get new tips if tip is free; 
        /// if tip is "pop", will drop pipette immediately. otherwise will do nothing.
        /// </para>
        /// <para>
        /// "get" removes picked tip from free_{left or right}_point
        /// </para>
        /// <para>
        /// if pipette is 1, left pipette will do operation. if 2, the right. default is right
        /// </para>
        /// </summary>
        /// <param name="point"></param>
        /// <param name="direction"></param>
        /// <returns></returns>
        private async Task dispenser_MovL(string point, string tip = "none", byte pipette = 2)
        {
            DKPoint dispenser_pt = new DKPoint();
            dispenser_conf.Points.TryGetValue(point, out dispenser_pt);
            if (pipette == 1)
            {
                dispenser_pt.x -= 4000;
            }
            await dispenserForm.MovL_hor(dispenser_pt);

            if (pipette == 1 && tip == "get" && free_right_tips.Contains(point))
            {
                dispenser_pt.lz = 135000;
                int wait_time = dispenserForm.MovL_ver(dispenser_pt, 0, 1).Result;
                await dispenserForm.MovL_ver(dispenser_pt, wait_time, 1);
                free_right_tips.Remove(point);
                dispenser_pt.lz = 0;
                await dispenserForm.MovL_ver(dispenser_pt, wait_time, 1);
                dispenser_pt.x += 4000;
            }
            if (pipette == 2 && tip == "get" && free_right_tips.Contains(point))
            {
                dispenser_pt.rz = 135000;
                int wait_time = dispenserForm.MovL_ver(dispenser_pt, 0, 2).Result;
                await dispenserForm.MovL_ver(dispenser_pt, wait_time, 2);
                free_right_tips.Remove(point);
                dispenser_pt.rz = 0;
                await dispenserForm.MovL_ver(dispenser_pt, wait_time, 2);
            }

            else if (tip == "pop")
            {
                dispenserForm.back_tip(pipette);
                running = 0;
            }
            else
            {
                return;
            }

        }

        /// <summary>
        /// <para> 
        /// pass in point name(string);
        /// </para> 
        /// <para>
        /// if pump is "spit", dispenser will descend to the coater and spit specified volume(ul) of liquid.
        /// if pump is "suck", dispenser will descend to the reagent bottles and suck specified volume(ul) of reagent.
        /// if pump is "descend", dispenser will go down to the coater only.
        /// if pump is "ascend", dispenser will go up to z=0.
        /// </para>
        /// <para>
        /// the default volume is 10 ul, in the experiment case, volume should be indicated by reagent Dictionary.
        /// can only use right tip, if no pipette on pump, will do nothing.
        /// </para>
        /// <para>
        /// if tip is 1, left pump will do operation. if tip is 2, the right. default is 2.
        /// </para>
        /// </summary>
        /// <param name="point"></param>
        /// <param name="tip"></param>
        /// <returns></returns>
        private async Task dispenser_pump(string point, string pump = "none", int volume = 10, byte tip = 2)
        {
            //int pipette_available = dispenserForm.check_pipette();

            DKPoint dispenser_pt = new DKPoint();
            dispenser_conf.Points.TryGetValue(point, out dispenser_pt);
            if (tip == 1)
            {
                dispenser_pt.x -= 4000;
            }

            await dispenserForm.MovL_hor(dispenser_pt);
            int wait_time0 = 0;

            if (tip == 1)
            {
                dispenser_pt.lz = 100000;//coater z
            }
            else if (tip == 2)
            {
                dispenser_pt.rz = 100000;//coater z
            }
            wait_time0 = dispenserForm.MovL_ver(dispenser_pt, 0, tip).Result;

            if (pump == "spit")
            {
                await dispenserForm.MovL_ver(dispenser_pt, wait_time0, tip);
                await dispenserForm.Tip_Spit(volume, tip);//right tip by default
                if (tip == 1)
                {
                    dispenser_pt.lz = 0;
                }
                else if (tip == 2)
                {
                    dispenser_pt.rz = 0;
                }
                await dispenserForm.MovL_ver(dispenser_pt, wait_time0, tip);
            }
            else if (pump == "suck")
            {
                if (tip == 1)
                {
                    dispenser_pt.lz = 175000;//bottle z
                }
                else if (tip == 2)
                {
                    dispenser_pt.rz = 175000;
                }
                int wait_time = dispenserForm.MovL_ver(dispenser_pt, 0, tip).Result;
                await dispenserForm.MovL_ver(dispenser_pt, wait_time, tip);
                await dispenserForm.Tip_Suck(volume, tip);//right tip by default
                if (tip == 1)
                {
                    dispenser_pt.lz = 0;
                }
                else if (tip == 2)
                {
                    dispenser_pt.rz = 0;
                }
                await dispenserForm.MovL_ver(dispenser_pt, wait_time, tip);
            }
            else if (pump == "descend")
            {
                await dispenserForm.MovL_ver(dispenser_pt, wait_time0, tip);
            }
            else if (pump == "ascend")
            {
                dispenser_pt.rz = 0;
                await dispenserForm.MovL_ver(dispenser_pt, wait_time0, tip);
            }

            if (tip == 1)
            {
                dispenser_pt.x += 4000;
            }
        }

        private void activate_timer(int round_index, int time, int volume)
        {
            running = 1;
            System.Timers.Timer timer = new System.Timers.Timer(time);
            timer.Elapsed += async (sender, e) =>
            {
                try
                {
                    await dispenser_pump("Coater", "spit", volume, 1);
                    await dispenser_MovL("Zero", "pop", 3);//pop all the tips
                    timer.Stop();
                    timer.Dispose();
                }
                catch (Exception ex)
                {
                    timer.Stop();
                    timer.Dispose();
                }
            };

            timer.Start();
        }

        private async void MoveTest_Click(object sender, EventArgs e)
        {
            //test platform experiment process
            try
            {
                //滑轨
                //FeedRail.connect("COM15", 9600);
                //FeedRail.change_state();
                //await Task.Delay(1000);
                //FeedRail.change_direction();
                //await Task.Delay(1000);
                //FeedRail.change_state();

                //光谱
                //specForm.read_in_situ_data(10000);

                //一轮
                //await round_test(1);

                //有几轮做几轮
                //for (int i = 1; i <= exp_rounds.Count; i++)
                //{
                //    await round_test(i);
                //}
            }
            catch (Exception ex)
            {
                Response.Text = ex.Message;
            }
        }

        /// <summary>
        /// load parameters for each experiment round, round = index + 1
        /// </summary>
        /// <param name="round">
        /// </param>
        /// <returns></returns>
        private async Task round_test(int round)//parametersL: round number
        {
            if (feature_count >= 3)
            {
                int index = round - 1;
                //coater: { speed, acceleration, duration }
                List<int> coater = new List<int> { exp_rounds[index][0], exp_rounds[index][1], exp_rounds[index][2] };//parameters: spin speed, acceleration, spin duration
                Log.Text += String.Format("round {0}: \n", round.ToString());
                for (int key = 0; key < reagent.Count; key++)
                {
                    Log.Text += String.Format("{0}, {1} ul at time {2}s; ", reagent.Keys.ElementAt(key), reagent.Values.ElementAt(key)[index].ToString(), dispense_time.Values.ElementAt(key)[index].ToString());
                }

                await arm_MovL("Zero");

                string substrate = free_substrates[0];//paramerter free_substrates[0]: position of the substrate to use

                if (reagent.Count() > 0)
                {
                    if (reagent.Count() == 1)
                    {
                        Thread thrd = new Thread(async () =>
                        {
                            await dispenser_MovL(free_right_tips[0], "get", 2);//parameter free_right_tips[0]: position of the tip for reagent
                        });
                        thrd.Start();

                        await arm_MovL(substrate, "pick at tray");
                        Response.Text = "Gripping start.";
                        await arm_MovL("Coater", "release at coater");
                        await arm_MovL("Calibrate");
                        await arm_MovL("Zero");

                        await dispenser_pump(reagent.Keys.ElementAt(0), "suck", reagent.Values.ElementAt(0)[index], 2);//parameter reagent.Keys.ElementAt(0): position of reagent bottle
                                                                                                                       //parameter reagent.Values.ElementAt(0)[index]: volume of reagent to suck this round
                        await dispenser_pump("Coater", "spit", reagent.Values.ElementAt(0)[index], 2);
                        await dispenser_MovL("Zero", "pop", 3);
                        Response.Text = "Dispensing liquid done. ";
                    }

                    else if (reagent.Count() == 2 && coater[2] > dispense_time.Values.ElementAt(1)[index])
                    {
                        Thread thrd = new Thread(async () =>
                        {
                            await dispenser_MovL(free_right_tips[0], "get", 2);
                            await dispenser_MovL(free_right_tips[0], "get", 1);
                        });
                        thrd.Start();

                        await arm_MovL(substrate, "pick at tray");
                        Response.Text = "Gripping start.";
                        await arm_MovL("Coater", "release at coater");
                        await arm_MovL("Calibrate");
                        await arm_MovL("Zero");

                        await dispenser_pump(reagent.Keys.ElementAt(0), "suck", reagent.Values.ElementAt(0)[index], 2);//parameter reagent.Keys.ElementAt(0): position of reagent bottle
                                                                                                                       //parameter reagent.Values.ElementAt(0)[index]: volume of reagent to suck
                        await dispenser_pump(reagent.Keys.ElementAt(1), "suck", reagent.Values.ElementAt(1)[index], 1);//parameter reagent.Keys.ElementAt(0): position of reagent bottle
                                                                                                                       //parameter reagent.Values.ElementAt(0)[index]: volume of reagent to suck

                        await dispenser_pump("Coater", "spit", reagent.Values.ElementAt(0)[index], 2);
                        await dispenser_pump("Coater", "descend", 0, 1);
                        Response.Text = "Dispensing reagent done. Waiting to dispense antisolvent";

                        activate_timer(index, dispense_time.Values.ElementAt(1)[index], reagent.Values.ElementAt(1)[index]);
                    }
                    //specForm.read_in_situ_data(coater[2]);
                    await coaterForm.Spin_Coat(coater[0], coater[1], coater[2]);
                }
                else
                {
                    await coaterForm.Spin_Coat(coater[0], coater[1], coater[2]);
                }
                Response.Text = "Coating done. ";

                while (running == 1)
                {
                    await Task.Delay(100);
                }

                await arm_MovL("Calibrate");
                await arm_MovL("Coater", "pick at coater");
                await arm_MovL(substrate, "release at tray");
                Response.Text = "Gripping done. ";

                await arm_MovL("Zero");
                Response.Text = "Moving done. ";
            }
        }

        // ai control
        /// <summary>
        /// starts ai control. when it is initiallizing, the platform will first check mqtt connection with broker to make sure it
        /// can successfully receive data from ai agent. connection can be secured in SpecForm
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private async void AI_Agent_Click(object sender, EventArgs e)
        {
            if (AI_Agent.BackColor == Color.Red)
            {
                if (Mqtt_connection.Connection_check(client))
                {
                    AI_Agent.BackColor = Color.Green;
                    AI_Agent.Text = "ai agent on";
                    Agent.clear_step_buffer();

                    Mqtt_connection.Subscribe(client, Agent.topic);
                    Response.Text = "client subscript to agent";

                    await Task.Run(async () =>
                    {
                        while (true)
                        {
                            if (Mqtt_connection.msg[0] == 'p')
                            {
                                if(Mqtt_connection.msg.Substring(1) == "start")// get step parameters from step_buffer
                                {
                                    for(int i = 0; i < Agent.step_buffer.Count; i++)
                                    {
                                        string[] parameters = Agent.step_buffer[i].Split(",");
                                        int spin_speed = Convert.ToInt32(parameters[0]);
                                        int spin_acc = Convert.ToInt32(parameters[1]);
                                        int spin_dur = Convert.ToInt32(parameters[2]);
                                        string reagent = parameters[3];
                                        int volume = Convert.ToInt32(parameters[4]);

                                        Response.Invoke(() =>
                                        {
                                            Response.Text += Agent.step_buffer[i] + " ";
                                        });
                                        await Task.Delay(3000);
                                        
                                        //await agent_round_test(spin_speed, spin_acc, spin_dur, reagent, volume);
                                    }
                                    Agent.clear_step_buffer();
                                }
                                else// save step parameters to step_buffer
                                {
                                    string parameters = Mqtt_connection.msg.Substring(1);
                                    Agent.to_step_buffer(parameters);
                                    Mqtt_connection.clear_msg();

                                    //Response.Invoke(() =>
                                    //{
                                    //    for (int i = 0; i < parameters.Length; i++)
                                    //    {
                                    //        Response.Text += parameters[i] + " ";
                                    //    }
                                    //});
                                }
                            }
                            if (AI_Agent.BackColor == Color.Red) break;
                        }
                    });
                }
            }
            else
            {
                if (Mqtt_connection.Connection_check(client))
                {
                    Mqtt_connection.Unsubscribe(client, Agent.topic);
                    Response.Text = "client unsubscript to agent";
                }
                else
                {
                    Response.Text = "client not connected";
                }
                AI_Agent.BackColor = Color.Red;
                AI_Agent.Text = "ai agent off";
                return;
            }
        }

        /// <summary>
        /// when ai agent handles the experiment, it can tell the platform to do rounds with a fixed routine and flexible parameters
        /// </summary>
        /// <param name="spin_speed">the speed of spinning in rpm</param>
        /// <param name="spin_acc">the acceleration speed of spinning</param>
        /// <param name="spin_dur">the duration of spinning</param>
        /// <param name="reagent">the position of the reagent to use, exp: "BP24".  
        /// the reagent corresponding to a certain point should be preset, and can later be found in reagent_layout.json</param>
        /// <param name="volume">the volume for the dispenser to suck and spit onto substrate</param>
        /// <returns></returns>
        private async Task agent_round_test(int spin_speed, int spin_acc, int spin_dur, string reagent, int volume)
        {
            await arm_MovL("Zero");

            string substrate = free_substrates[0];

            Thread thrd = new Thread(async () =>
            {
                await dispenser_MovL(free_right_tips[0], "get", 2);
            });
            thrd.Start();

            await arm_MovL(substrate, "pick at tray");
            //Response.Text = "Gripping start.";
            await arm_MovL("Coater", "release at coater");
            await arm_MovL("Calibrate");
            await arm_MovL("Zero");

            await dispenser_pump(reagent, "suck", volume, 2);
            await dispenser_pump("Coater", "spit", volume, 2);
            await dispenser_MovL("Zero", "pop", 3);
            //Response.Text = "Dispensing liquid done. ";

            await coaterForm.Spin_Coat(spin_speed, spin_acc, spin_dur);
            //specForm.read_in_situ_data(spin_dur);
            //Response.Text = "Coating done. ";

            while (running == 1)
            {
                await Task.Delay(100);
            }

            await arm_MovL("Calibrate");
            await arm_MovL("Coater", "pick at coater");
            await arm_MovL(substrate, "release at tray");
            //Response.Text = "Gripping done. ";

            await arm_MovL("Zero");
            //Response.Text = "Moving done. ";
  
        }
    }
}
