<?xml version="1.0" encoding="utf-8" ?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="generator" content="Docutils 0.12: http://docutils.sourceforge.net/" />
<title></title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-MrcW6ZMFYlzcLA8Nl+NtUVF0sA7MsXsP1UyJoMp4YLEuNSfAP+JcXn/tWtIaxVXM" crossorigin="anonymous"></script>
<script src="https://code.jquery.com/jquery-3.6.0.min.js" integrity="sha256-/xUj+3OJU5yExlq6GSYGSHk7tPXikynS7ogEvDej/m4=" crossorigin="anonymous"></script>
<style type="text/css">
.sub-report-title {}
.run-title {}

h1 { padding-top: 35px; }
h2 { padding-top: 20px; }
h3 { padding-top: 15px; }

.elem-desc {}
.elem-caption {
    margin-top: 15px
    margin-bottom: 0;
}
.elem-filename {}

div.elem-image {
  width: 100%;
  page-break-before:always;
}

.elem-image object.svg-reportlet {
    width: 100%;
    padding-bottom: 5px;
}
body {
    padding: 65px 10px 10px;
}

.boiler-html {
    font-family: "Bitstream Charter", "Georgia", Times;
    margin: 20px 25px;
    padding: 10px;
    background-color: #F8F9FA;
}

div#boilerplate pre {
    margin: 20px 25px;
    padding: 10px;
    background-color: #F8F9FA;
}

#errors div, #errors p {
    padding-left: 1em;
}
</style>
</head>
<body>


<nav class="navbar fixed-top navbar-expand-lg navbar-light bg-light">
<div class="collapse navbar-collapse">
    <ul class="navbar-nav">
    {% for sub_report in sections %}
        {% if sub_report.isnested %}
        <li class="nav-item dropdown">
            <a class="nav-link dropdown-toggle" id="navbar{{ sub_report.name }}" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false" href="#">{{ sub_report.name }}</a>
            <div class="dropdown-menu" aria-labelledby="navbar{{ sub_report.name }}">
                {% for run_report in sub_report.reportlets %}
                    {% if run_report.title %}
                    <a class="dropdown-item" href="#{{run_report.name}}">{{run_report.title}}</a>
                    {% endif %}
                {% endfor %}
            </div>
        </li>
        {% else %}
        <li class="nav-item"><a class="nav-link" href="#{{sub_report.name}}">{{sub_report.name}}</a></li>
        {% endif %}
    {% endfor %}
        <li class="nav-item"><a class="nav-link" href="#boilerplate">Methods</a></li>
        <li class="nav-item"><a class="nav-link" href="#errors">Errors</a></li>
    </ul>
</div>
</nav>
<noscript>
    <h1 class="text-danger"> The navigation menu uses Javascript. Without it this report might not work as expected </h1>
</noscript>

{% for sub_report in sections %}
    {% if sub_report.reportlets %}
    <div id="{{ sub_report.name }}">
    <h1 class="sub-report-title">{{ sub_report.name }}</h1>
    {% for run_report in sub_report.reportlets %}
        <div id="{{run_report.name}}">
            {% if run_report.title %}<h2 class="sub-report-group">{{ run_report.title }}</h2>{% endif %}
            {% if run_report.subtitle %}<h3 class="run-title">{{ run_report.subtitle }}</h3>{% endif %}
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

<div id="boilerplate">
    <h1 class="sub-report-title">Methods</h1>
    {% if boilerplate %}
    <p>We kindly ask to report results preprocessed with this tool using the following
       boilerplate.</p>
    <ul class="nav nav-tabs" id="myTab" role="tablist">
        {% for b in boilerplate %}
        <li class="nav-item">
            <a class="nav-link {% if b[0] == 0 %}active{% endif %}" id="{{ b[1] }}-tab" data-toggle="tab" href="#{{ b[1] }}" role="tab" aria-controls="{{ b[1] }}" aria-selected="{% if b[0] == 0 %}true{%else%}false{% endif %}">{{ b[1] }}</a>
        </li>
        {% endfor %}
    </ul>
    <div class="tab-content" id="myTabContent">
      {% for b in boilerplate %}
      <div class="tab-pane fade {% if b[0] == 0 %}active show{% endif %}" id="{{ b[1] }}" role="tabpanel" aria-labelledby="{{ b[1] }}-tab">{{ b[2] }}</div>
      {% endfor %}
    </div>
    {% else %}
    <p class="text-danger">Failed to generate the boilerplate</p>
    {% endif %}
</div>

<div id="errors">
    <h1 class="sub-report-title">Errors</h1>
    {% for error in errors %}
    <details>
        <summary>Node Name: {{ error.node }}</summary><br>
        <div>
            File: <code>{{ error.file }}</code><br>
            Working Directory: <code>{{ error.node_dir }}</code><br>
            Inputs: <br>
            <ul>
            {% for name, spec in error.inputs %}
                <li>{{ name }}: <code>{{ spec }}</code></li>
            {% endfor %}
            </ul>
            <pre>{{ error.traceback }}</pre>
        </div>
    </details>
    {% else %}
    <p>No errors to report!</p>
    {% endfor %}
</div>


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
