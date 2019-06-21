namespace IFN647_PROJECT
{
    partial class welcome_page
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
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.beginbutton1 = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // label2
            // 
            this.label2.BackColor = System.Drawing.Color.White;
            this.label2.FlatStyle = System.Windows.Forms.FlatStyle.Popup;
            this.label2.Font = new System.Drawing.Font("Tahoma", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(38, 273);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(891, 356);
            this.label2.TabIndex = 10;
            this.label2.Text = "Welcome to use our EDU information system!\r\nThis system is easy to use\r\nif you ha" +
    "ve any problems, please click help button to get hints!!\r\n\r\nGOOD LUCK!!";
            // 
            // label1
            // 
            this.label1.Font = new System.Drawing.Font("Castellar", 16.2F, ((System.Drawing.FontStyle)((System.Drawing.FontStyle.Bold | System.Drawing.FontStyle.Italic))), System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Image = global::IFN647_PROJECT.Properties.Resources._240px_QUTLogo_svg;
            this.label1.ImageAlign = System.Drawing.ContentAlignment.TopLeft;
            this.label1.Location = new System.Drawing.Point(12, 37);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(904, 206);
            this.label1.TabIndex = 9;
            this.label1.Text = "EDU INFORMATION SYSTEM";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // beginbutton1
            // 
            this.beginbutton1.BackColor = System.Drawing.Color.White;
            this.beginbutton1.FlatAppearance.BorderSize = 0;
            this.beginbutton1.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.beginbutton1.Font = new System.Drawing.Font("Arial Rounded MT Bold", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.beginbutton1.ForeColor = System.Drawing.Color.Navy;
            this.beginbutton1.Image = global::IFN647_PROJECT.Properties.Resources.ok_2_icon1;
            this.beginbutton1.ImageAlign = System.Drawing.ContentAlignment.TopCenter;
            this.beginbutton1.Location = new System.Drawing.Point(339, 444);
            this.beginbutton1.Name = "beginbutton1";
            this.beginbutton1.Size = new System.Drawing.Size(196, 134);
            this.beginbutton1.TabIndex = 11;
            this.beginbutton1.Text = "BEGIN!!";
            this.beginbutton1.TextAlign = System.Drawing.ContentAlignment.BottomCenter;
            this.beginbutton1.UseVisualStyleBackColor = false;
            this.beginbutton1.Click += new System.EventHandler(this.beginbutton1_Click);
            // 
            // welcome_page
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.DodgerBlue;
            this.ClientSize = new System.Drawing.Size(995, 668);
            this.Controls.Add(this.beginbutton1);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Name = "welcome_page";
            this.Text = "welcome_page";
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button beginbutton1;
    }
}