from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import User, CitizenshipDetail


class UserRegistrationForm(UserCreationForm):
    """Form for user registration with password confirmation."""

    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name')
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Username'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Enter Email'}),
            'first_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter First Name'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Last Name'}),
        }
    password1 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={'class': 'form-control', 'autocomplete': 'new-password', 'placeholder': 'Enter Password'}))
    password2 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={'class': 'form-control', 'autocomplete': 'new-password', 'placeholder': 'Confirm Password'}))


class LoginForm(AuthenticationForm):
    """Form for user login."""
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'})
    )


class CitizenshipDetailForm(forms.ModelForm):
    """Form for submitting citizenship verification details."""

    class Meta:
        model = CitizenshipDetail
        fields = [
            'full_name', 'place_of_birth', 'district_of_birth', 'permanent_address_district',
            'municipality', 'ward_number', 'date_of_birth', 'father_name', 'father_address',
            'mother_name', 'mother_address', 'spouse_name', 'spouse_address', 'citizenship_number', 'front_image','back_image'
        ]
        widgets = {
            'full_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Full Name (नाम थर)'}),
            'place_of_birth': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Place of Birth (जन्म स्थान)'}),
            'district_of_birth': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'District of Birth (जिल्ला)'}),
            'permanent_address_district': forms.TextInput(
                attrs={'class': 'form-control', 'placeholder': 'Permanent Address District (जिल्ला)'}),
            'municipality': forms.TextInput(
                attrs={'class': 'form-control', 'placeholder': 'Municipality/Metropolitan City (म.न.पा./न.पा.)'}),
            'ward_number': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Ward Number (वडा नं.)'}),
            'date_of_birth': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'father_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Father’s Name (बाबुको नाम थर)'}),
            'father_address': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Father’s Address (ठेगाना)'}),
            'mother_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Mother’s Name (आमाको नाम थर)'}),
            'mother_address': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Mother’s Address (ठेगाना)'}),
            'spouse_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Spouse’s Name (पति/पत्नीको नाम थर)'}),
            'spouse_address': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Spouse’s Address (ठेगाना)'}),
            'citizenship_number': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Citizenship Number'}),
        }


class CitizenshipForm(forms.Form):
    front_photo = forms.ImageField(required=True)
    back_photo = forms.ImageField(required=True)