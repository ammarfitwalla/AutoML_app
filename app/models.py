from django.db import models
from django.contrib.auth.models import User
from picklefield.fields import PickledObjectField


# Create your models here.


def user_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT / user_<id>/<filename>
    return 'user_{0}/{1}'.format(instance.user.id, filename)


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    email_confirmed = models.BooleanField(default=False)

    def __str__(self):
        return str(self.user)


class Document(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    document = models.ImageField(upload_to='documents/', null=True, unique=False)
    uploaded_at = models.DateTimeField(auto_now_add=True, null=True)

    def __str__(self):
        return str(self.document)

class TrainedModels(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    model_name = models.CharField(max_length=255, blank=False)
    model_type = models.CharField(max_length=255, blank=False)
    project_name = models.CharField(max_length=255, blank=False)
    column_names = models.CharField(max_length=1000, blank=False)
    model_file = PickledObjectField()

    def __str__(self):
        return str(self.project_name)
