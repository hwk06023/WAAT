from django.db import models

# Create your models here.

class Hyuser(models.Model):
    username = models.CharField(max_length=64,
                                verbose_name='사용자명')

    password = models.CharField(max_length=64,
                                verbose_name='비밀번호') 

    registered_dttm = models.DateTimeField(auto_now_add=True,
                                           verbose_name='등록시간')

    class Meta:
        db_table = 'HYuniv_hyuser'
