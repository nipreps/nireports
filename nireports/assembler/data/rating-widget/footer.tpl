<div id="{{ config.id }}-menu" class="card position-fixed d-none" style="{{ config.settings.style }}">
<div class="card-header m-0">
    {{ config.settings.navbar_label }}
    <button type="button" class="btn-close position-absolute top-0 end-0" aria-label="Close" id="close-{{ config.id }}-menu" onclick="toggle_rating()" style="margin: 10px 10px 0 0"></button>
</div>
<div class="card-body">
<div class="accordion">
  <div class="accordion-item">
    <h2 class="accordion-header" id="{{ config.components.slider.id }}-head">
      <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#{{ config.components.slider.id }}-collapse" aria-expanded="true" aria-controls="{{ config.components.slider.id }}-collapse">{{ config.components.slider.title }}</button>
    </h2>
    <div id="{{ config.components.slider.id }}-collapse" class="accordion-collapse collapse show" aria-labelledby="{{ config.components.slider.id }}-head">
      <div class="accordion-body">
        <input type="range" min="{{ config.components.slider.settings.min }}" max="{{ config.components.slider.settings.max }}" step="{{ config.components.slider.settings.step }}" value="{{ config.components.slider.settings.value }}" id="{{ config.components.slider.id }}" class="slider">
        <ul class="list-group list-group-horizontal slider-labels" style="width: 100%">
            {% for opt in config.components.slider.options %}
            <li class="list-group-item list-group-item-{{ opt[1] }} small" style="font-size: 0.7em; width: 25%; text-align:center">{{ opt[0] }}</li>
            {% endfor %}
        </ul>
      </div>
    </div>
  </div>

  {% if config.components.artifacts %}
  <div class="accordion-item">
    <h2 class="accordion-header" id="{{ config.components.artifacts.id }}-head">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#{{ config.components.artifacts.id }}-collapse" aria-expanded="false" aria-controls="{{ config.components.artifacts.id }}-collapse">
        {{ config.components.artifacts.title }}
      </button>
    </h2>
    <div id="{{ config.components.artifacts.id }}-collapse" class="accordion-collapse collapse" aria-labelledby="{{ config.components.artifacts.id }}-head">
      <div class="accordion-body">
        <fieldset id="{{ config.components.artifacts.id }}-group" class="form-group">
            {% for name, label in config.components.artifacts.options.items() %}
            <div class="form-check form-switch small">
                <input class="form-check-input" type="checkbox" name="{{ name }}" id="{{ config.components.artifacts.id }}-item-{{ loop.index0 }}" />
                <label class="form-check-label" for="{{ config.components.artifacts.id }}-item-{{ loop.index0 }}">{{ label }}</label>
            </div>
            {% endfor %}
        </fieldset>
      </div> <!-- accordion-body -->
    </div> <!-- accordion-collapse -->
  </div> <!-- accordion-item -->
  {% endif %}
  {% if config.components.extra %}
  <div class="accordion-item">
    <h2 class="accordion-header" id="{{ config.components.extra.id }}-head">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#{{ config.components.extra.id }}-collapse" aria-expanded="false" aria-controls="{{ config.components.extra.id }}-collapse">
        {{ config.components.extra.title }}
      </button>
    </h2>
    <div id="{{ config.components.extra.id }}-collapse" class="accordion-collapse collapse" aria-labelledby="{{ config.components.extra.id }}-head">
      <div class="accordion-body">
        <div class="input-group">
          <span class="input-group-text">{{ config.components.extra.components[0].settings.label }}</span>
          <textarea class="form-control" aria-label="{{ config.components.extra.components[0].settings.label }}" id="{{ config.components.extra.id }}-comments"></textarea>
        </div>

        <p style="margin-top: 20px; font-weight: bold">{{ config.components.extra.components[1].settings.label }}</p>
        <input type="range" min="{{ config.components.extra.components[1].settings.min }}" max="{{ config.components.extra.components[1].settings.max }}" step="{{ config.components.extra.components[1].settings.step }}" value="{{ config.components.extra.components[1].settings.value }}" id="{{ config.components.extra.id }}-confidence" class="slider" style="margin-left: 22%;width: 56%;">
        <ul class="list-group list-group-horizontal slider-labels" style="width: 100%">
            {% for opt in config.components.extra.components[1].options %}
            <li class="list-group-item list-group-item-{{ opt[1] }} small" style="width: 50%; text-align:center">{{ opt[0] }}</li>
            {% endfor %}
        </ul>
       </div> <!-- accordion-body -->
    </div> <!-- accordion-collapse -->
  </div> <!-- accordion-item -->
  {% endif %}
</div>
<div style="margin-top: 10px">
<a class="btn btn-primary disabled" id="{{ config.components.actions[0].id }}" href="{{ config.components.actions[0].href }}">{{ config.components.actions[0].text }}</a>
{% if metadata.access_token != "<secret_token>" %}
<button class="btn btn-primary" id="{{ config.components.actions[1].id }}" value="{{ metadata.access_token }}" disabled>{{ config.components.actions[1].text }}</button>
{% endif %}
</div>
<script type="text/javascript">
var MINIMUM_RATING_TIME = {{ config.settings.mintime }}
$('#{{ config.components.slider.id }}').on('input', function() {

    if ( (Date.now() - timestamp) / 1000 > MINIMUM_RATING_TIME) {
        $('#{{ config.components.actions[0].id }}').removeClass('disabled');
        $('#{{ config.components.actions[0].id }}').removeAttr('aria-disabled');
        $('#{{ config.components.actions[1].id }}').removeAttr('disabled');
    };

    $('#{{ config.components.slider.id }}-collapse .list-group-item').removeClass(function(index, classname) {
        return (classname.match(/(^|\s)bg-\S+/g) || []).join(' ');
    });
    $('#{{ config.components.slider.id }}-collapse .list-group-item').removeClass(function(index, classname) {
        return (classname.match(/(^|\s)text-\S+/g) || []).join(' ');
    });

    if ( $(this).val() < 1.5 ) {
        $('#{{ config.components.slider.id }}-collapse .list-group-item-danger').addClass('bg-danger text-white');
    } else if ( $(this).val() > 3.5 ) {
        $('#{{ config.components.slider.id }}-collapse .list-group-item-success').addClass('bg-success text-white');
    } else if ( $(this).val() < 2.5 ) {
        $('#{{ config.components.slider.id }}-collapse .list-group-item-warning').addClass('bg-warning text-dark');
    } else {
        $('#{{ config.components.slider.id }}-collapse .list-group-item-primary').addClass('bg-primary text-white');
    };

    var payload = read_form();
});

$('#{{ config.components.extra.id }}-confidence').on('input', function() {
    if ( (Date.now() - timestamp) / 1000 > MINIMUM_RATING_TIME) {
        $('#{{ config.components.actions[0].id }}').removeClass('disabled');
        $('#{{ config.components.actions[0].id }}').removeAttr('aria-disabled');
        $('#{{ config.components.actions[1].id }}').removeAttr('disabled');
    };

    $('#{{ config.components.extra.id }}-collapse .list-group-item').removeClass(function(index, classname) {
        return (classname.match(/(^|\s)bg-\S+/g) || []).join(' ');
    });
    $('#{{ config.components.extra.id }}-collapse .list-group-item').removeClass(function(index, classname) {
        return (classname.match(/(^|\s)text-\S+/g) || []).join(' ');
    });

    if ( $(this).val() < 2.0 ) {
        $('#{{ config.components.extra.id }}-collapse .list-group-item-warning').addClass('bg-warning text-dark');
    } else {
        $('#{{ config.components.extra.id }}-collapse .list-group-item-success').addClass('bg-success text-white');
    };

    var payload = read_form();
});


$('#{{ config.components.extra.id }}-comments').bind('input propertychange', function() {
    if ( (Date.now() - timestamp) / 1000 > MINIMUM_RATING_TIME) {
        $('#{{ config.components.actions[0].id }}').removeClass('disabled');
        $('#{{ config.components.actions[0].id }}').removeAttr('aria-disabled');
        $('#{{ config.components.actions[1].id }}').removeAttr('disabled');
    };
});

$( '#{{ config.components.actions[1].id }}' ).click( function() {
    var payload = read_form();
    var md5sum = "{{ metadata.md5sum }}";
    var params = {
        'rating': payload['rating'],
        'md5sum': md5sum,
        'name': "",
        'comment': JSON.stringify(payload['artifacts'])
    };

    // disable development releases
    var authorization = $(this).val();
    var ratingReq = new XMLHttpRequest();
    ratingReq.open("POST", "{{ metadata.endpoint }}");
    ratingReq.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    ratingReq.setRequestHeader("Authorization", authorization);
    ratingReq.onload = function () {
        status = ratingReq.status;
        $('#{{ config.components.actions[1].id }}').removeClass('btn-primary');
        $('#{{ config.components.actions[1].id }}').attr('disabled', true);
        $('#{{ config.components.actions[1].id }}').attr('aria-disabled', true);
        $('#{{ config.components.actions[1].id }}').prop('disabled');
        $('#{{ config.components.actions[1].id }}').addClass('disabled');
        $('#{{ config.components.actions[1].id }}').removeClass('active');
        if (status === "201") {
            $('#{{ config.components.actions[1].id }}').addClass('btn-success');
            $('#{{ config.components.actions[1].id }}').html('Posted!');
        } else {
            $('#{{ config.components.actions[1].id }}').addClass('btn-danger');
            $('#{{ config.components.actions[1].id }}').html('Failed');
        };
    };
    ratingReq.send(JSON.stringify(params));
});

$( 'body' ).on( 'click', '#{{ config.components.artifacts.id }}-group input', function(e) {
    if ( (Date.now() - timestamp) / 1000 > MINIMUM_RATING_TIME) {
        $('#{{ config.components.actions[0].id }}').removeClass('disabled');
        $('#{{ config.components.actions[0].id }}').removeAttr('aria-disabled');
        $('#{{ config.components.actions[1].id }}').removeAttr('disabled');
    };
    
    var payload = read_form();
});

$( 'body' ).on( 'click', '#{{ config.id }}-toggler', function(e) {
    toggle_rating();
});
</script>
</div>