using uPLibrary.Networking.M2Mqtt;
using uPLibrary.Networking.M2Mqtt.Exceptions;
using System.Threading;
using System.Net.Sockets;
using System.Net;
using System.Windows.Forms;
using WinFormsApp_Draft.Auto;
using System.Threading.Tasks;

namespace WinFormsApp_Draft
{
    public partial class SpecForm : Form
    {
        static string broker = "192.168.120.129";
        const int port = 1883;
        public static MqttClient client = new MqttClient(broker, port, false, MqttSslProtocols.None, null, null);

        //topics to publish: control, counts, wavelength, time
        string clientId = "123abc";
        string username = "platform";
        string password = "s208ht";

        private Socket spectrum_Client;

        public SpecForm()
        {
            InitializeComponent();
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            Connect_Brocker.Text = "connect broker";
            Broker_addr.Text = broker;
            Message.Text = "test message";
        }

        private void Connect_Brocker_Click(object sender, EventArgs e)
        {
            if (broker != Broker_addr.Text)
            {
                broker = Broker_addr.Text;
                client = new MqttClient(broker, port, false, MqttSslProtocols.None, null, null);
            }

            try
            {
                if (Connect_Brocker.Text == "connect broker")
                {
                    Response0.Text = "1";
                    client = Mqtt_connection.ConnectMQTT(client, clientId, username, password);
                    Mqtt_connection.Subscribe(client, "control");

                    if (Mqtt_connection.Connection_check(client))
                    {
                        Connect_Brocker.Text = "disconnect broker";
                        Response0.Text = "broker connected";
                    }
                }
                else if (Connect_Brocker.Text == "disconnect broker")
                {
                    client = Mqtt_connection.DisconnectMQTT(client);
                    Mqtt_connection.Unsubscribe(client, "control");
                    if (!Mqtt_connection.Connection_check(client))
                    {
                        Connect_Brocker.Text = "connect broker";
                        Response0.Text = "broker disconnected";
                    }
                }

            }
            catch (Exception ex)
            {
                Response0.Text = ex.Message;
            }
        }
        private async void Connect_Spectrum_Click(object sender, EventArgs e)
        {
            try
            {
                await Task.Run(() =>
                {
                    if (Connect_Spectrum.Text == "connect spectrum")
                    {
                        IPAddress ip = IPAddress.Parse("127.0.0.1");
                        int port = 1701;
                        spectrum_Client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                        spectrum_Client.Connect(new IPEndPoint(ip, port));
                        if (spectrum_Client.Connected)
                        {
                            Connect_Spectrum.Text = "disconnect spectrum";
                            Response0.Text = "spectrum connected";
                        }
                    }
                    else if (Connect_Spectrum.Text == "disconnect spectrum")
                    {
                        spectrum_Client.Disconnect(false);
                        if (!spectrum_Client.Connected)
                        {
                            Connect_Spectrum.Text = "connect spectrum";
                            Response0.Text = "spectrum disconnected";
                        }
                    }
                });
            }
            catch (Exception ex)
            {
                Response0.Text = ex.Message;
            }
        }
        private void DisconnectPyClient_Click(object sender, EventArgs e)
        {
            string msg_quit = "quit";
            Mqtt_connection.Publish(client, "control", msg_quit);
        }

        private async void Publish_Click(object sender, EventArgs e)
        {
            await Task.Run(() =>
            {
                read_in_situ_data(10000);
            });
        }

        /// <summary>
        /// repeatingly request for spectrum data within specified spin duration.
        /// default interval is 100ms.
        /// 
        /// </summary>
        /// <param name="spin_duration"></param>
        /// <param name="interval"></param>
        public async void read_in_situ_data(int spin_duration, int interval = 100)
        {
            string msg_continue = "continue";
            Mqtt_connection.Publish(client, "control", msg_continue);
            await Mqtt_connection.wait();

            SendProgress.Invoke(() =>
            {
                SendProgress.Maximum = spin_duration;
                SendProgress.Minimum = 0;
                SendProgress.Step = interval;
                SendProgress.Value = 0;
            });
            int ms_sum = 0;
            System.Timers.Timer timer = new System.Timers.Timer();
            timer.Interval = interval;
            timer.AutoReset = true;
            timer.Elapsed += (sender, e) =>
            {
                ms_sum += interval;
                record_data(ms_sum);

                SendProgress.Invoke(new Action(() =>
                {
                    SendProgress.PerformStep();
                }));
                if (ms_sum == spin_duration - interval)
                {
                    timer.AutoReset = false;
                }
                else if (ms_sum == spin_duration)
                {
                    string msg_stop = "stop";
                    Mqtt_connection.Publish(client, "control", msg_stop);
                }
            };
            timer.Start();
        }

        /// <summary>
        /// obtain data from spectrometer, and then send them to data analyzing client through server
        /// </summary>
        private void read_spectrum()
        {
            Scan.do_single_scan(spectrum_Client);
            List<string> data = Data.get_data(spectrum_Client);

            string msg_wl = data[0];
            Mqtt_connection.Publish(client, "wavelength", msg_wl);
            string msg_cts = data[1];
            Mqtt_connection.Publish(client, "counts", msg_cts);
        }

        //pseudo spectrum data
        private void read_pseudo_spectrum(int ms)
        {
            string msg0 = "";
            string msg1 = "";
            for (int j = 1; j < 1000; j++)
            {
                msg0 += " " + ( ms + j).ToString();
                msg1 += " " + j.ToString();
            }
            Mqtt_connection.Publish(client, "counts", msg0);
            Mqtt_connection.Publish(client, "wavelength", msg1);
        }

        /// <summary>
        /// get current time and control the spectrometer to record the corresponding frame of data, then send them to data analyzing client
        /// </summary>
        /// <param name="ms"></param>
        private async void record_data(int ms)
        {
            string msg_time = ms.ToString();
            Mqtt_connection.Publish(client, "time", msg_time);
            read_pseudo_spectrum(ms);
            //read_spectrum();
            string msg_record = "record";
            Mqtt_connection.Publish(client, "control", msg_record);
            await Mqtt_connection.wait();
        }
    }
}
