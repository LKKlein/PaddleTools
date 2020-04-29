import requests

from paddletools import logger


class WeChatReminder(object):

    def __init__(self, secret):
        super(WeChatReminder, self).__init__()
        self.sess = requests.Session()
        a = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
        self.sess.mount("http://", a)
        self.sess.mount("https://", a)
        self.url = "https://sc.ftqq.com/{secret}.send".format(secret=secret)

    def send(self, title, content="", retry=3):
        if len(title) > 256:
            logger.warning("The title should no longer than 256 words!")
            logger.warning("The title will be truncated!")
            title = title[:256]
        if len(content) > 65536:
            logger.warning("The content should no longer than 65536 words!")
            content = content[:65536]

        data = {
            "text": title,
            "desp": content
        }
        for _ in range(retry):
            try:
                response = self.sess.get(self.url, params=data, timeout=(2, 2))
                code = response.status_code
                assert code == 200, "request error! status code: {}".format(code)
                res = response.text
                if "success" in res:
                    logger.info("message send successfully!")
                else:
                    logger.warning("something wrong! response: {}".format(res))
                break
            except Exception as e:
                logger.error("request error happend! {}".format(e))
