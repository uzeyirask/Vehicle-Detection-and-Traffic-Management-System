using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Data.SqlClient;


namespace Tasarim2Form
{
    public partial class Form1 : Form
    {
        // Bağlantı dizesini tam ve doğru olarak belirtin
        private string connectionString = "Server=MUA-PC\\SQLEXPRESS;Initial Catalog=ArabaSayaciDB;Integrated Security=True;";

        public Form1()
        {
            InitializeComponent();
        }

        private void sorguGost(string veriler)      //Sorgu Icin yordam
        {
            // SqlDataAdapter ve DataSet kullanarak verileri getirin
            using (SqlConnection con = new SqlConnection(connectionString))
            {
                SqlDataAdapter da = new SqlDataAdapter(veriler, con);
                DataSet ds = new DataSet();
                da.Fill(ds);
                dataGridView1.DataSource = ds.Tables[0];
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            // ArabaSayisi tablosundaki tüm verileri getir
            sorguGost("SELECT * FROM ArabaSayisi");
        }

        private void button2_Click(object sender, EventArgs e)
        {
            // Kameralar tablosundaki tüm verileri getir
            sorguGost("SELECT * FROM Kameralar");
        }
    }
}
