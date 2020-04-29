import smtplib
from email.header import Header
from email.mime.text import MIMEText

from paddletools import logger
from paddletools.config import email_stmp_server


class EmailReminder(object):

    def __init__(self, sender, receiver, password):
        super(EmailReminder, self).__init__()
        self.receiver = receiver
        self.sender = sender
        self.password = password
        email_domain = sender.split("@")[1]
        if email_domain not in email_stmp_server:
            raise Exception("{} is not supported now! please change another send email".format(
                email_domain))
        server_info = email_stmp_server[email_domain]
        self.server = server_info["server"]
        self.port = server_info["port"]
        self.use_ssl = server_info["use_ssl"]

    def send(self, title, content, retry=3):
        for _ in range(retry):
            try:
                msg = MIMEText(content, "plain", "utf-8")
                msg['From'] = Header(self.sender)
                msg['To'] = Header(self.receiver)
                msg['Subject'] = Header(title)

                if not self.use_ssl:
                    server = smtplib.SMTP(self.server, self.port)
                else:
                    server = smtplib.SMTP_SSL(self.server, self.port)
                server.login(self.sender, self.password)
                server.sendmail(self.sender, [self.receiver], msg.as_string())
                server.quit()
                logger.info("email reminder send successfully!")
                break
            except Exception as e:
                logger.error("email send error! {}".format(e))
