<?xml version="1.0" encoding="utf-8" ?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="generator" content="NiReports: https://www.nipreps.org/" />
<title>{{ title }}</title>
<script src="https://code.jquery.com/jquery-3.6.0.min.js" integrity="sha256-/xUj+3OJU5yExlq6GSYGSHk7tPXikynS7ogEvDej/m4=" crossorigin="anonymous"></script>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-kenU1KFdBIe4zVF0s0G1M5b4hcpxyD9F7jL+jjXkk+Q2h455rYXK/7HAuoJl+0I4" crossorigin="anonymous"></script>

{% for hh in header %}
{{ hh }}
{% endfor %}
</head>
<body style="font-family: helvetica;">
<nav class="navbar fixed-top navbar-expand-lg bg-light">
<div class="container-fluid">
<div class="collapse navbar-collapse" id="navbarSupportedContent">
    <ul class="navbar-nav me-auto mb-2 mb-lg-0">
    {% for sub_report in sections %}
        {% if sub_report.isnested %}
        <li class="nav-item dropdown">
            <a class="nav-link dropdown-toggle" id="navbar{{ sub_report.name }}" role="button" data-bs-toggle="dropdown" aria-expanded="false" href="#{{sub_report.name}}">
            {{ sub_report.name }}
            </a>
            <ul class="dropdown-menu">
            {% for run_report in sub_report.reportlets %}
                {% if run_report.title %}
                <li><a class="dropdown-item" href="#{{run_report.name}}">{{run_report.title}}</a></li>
                {% endif %}
            {% endfor %}
            </ul>
        </li>
        {% else %}
        <li class="nav-item"><a class="nav-link" href="#{{sub_report.name}}">{{sub_report.name}}</a></li>
        {% endif %}
    {% endfor %}
    </ul>
</div>
</div>
{% if navbar %}
<div class="d-flex flex-row-reverse">
{% for item in navbar %}
{{ item }}
{% endfor %}
</div>
{% endif %}
</nav>
<noscript>
    <h1 class="text-danger"> The navigation menu uses Javascript. Without it this report might not work as expected </h1>
</noscript>

{% for sub_report in sections %}
    {% if sub_report.reportlets %}
    <div id="{{ sub_report.name }}" class="mt-5">
    <h1 class="sub-report-title pt-5 ps-4">{{ sub_report.name }}</h1>
    {% for run_report in sub_report.reportlets %}
        <div id="{{run_report.name}}" class="ps-4 pe-4 mb-2">
            {% if run_report.title %}<h2 class="sub-report-group mt-4">{{ run_report.title }}</h2>{% endif %}
            {% if run_report.subtitle %}<h3 class="run-title mt-3">{{ run_report.subtitle }}</h3>{% endif %}
            {% if run_report.description %}<p class="elem-desc">{{ run_report.description }}</p>{% endif %}
            {% for elem in run_report.components %}
                {% if elem[0] %}
                    {% if elem[1] %}<p class="elem-caption">{{ elem[1] }}</p>{% endif %}
                    {{ elem[0] }}
                {% endif %}
            {% endfor %}
        </div>
    {% endfor %}
    </div>
    {% endif %}
{% endfor %}

{% for ff in footer %}
{{ ff }}
{% endfor %}

<script type="text/javascript">
function toggle(id) {
    var element = document.getElementById(id);
    if(element.style.display == 'block')
        element.style.display = 'none';
    else
        element.style.display = 'block';
}
</script>
</body>
</html>
