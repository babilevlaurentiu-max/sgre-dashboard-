load("dashboard_objects.RData")
library(shiny)
library(bslib)
library(dplyr)
library(recipes)
library(xgboost)
library(tibble)
library(ggplot2)
library(DT)
library(scales)
library(forcats)

required_objects <- c(
  "df_clean",
  "xgb_results_global_wbs_raw",
  "xgb_models_global_wbs",
  "xgb_results_region_platform_wbs_raw",
  "xgb_models_region_platform_wbs"
)

missing_objects <- required_objects[!vapply(required_objects, exists, logical(1), envir = .GlobalEnv)]

if (length(missing_objects) > 0) {
  stop(
    paste0(
      "These objects are missing from the current R session: ",
      paste(missing_objects, collapse = ", "),
      ". Run Master analysis V3.R first."
    ),
    call. = FALSE
  )
}

fmt_num <- scales::label_number(accuracy = 1, big.mark = ",")

rmse_vec_local <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

valid_region_levels <- c("SEA", "NEME", "LATAM", "APAC", "NAM")
min_train_threshold <- 80

df_model_base_dashboard <- df_clean %>%
  mutate(
    wbs_grouping = if_else(is.na(wbs_grouping) | trimws(wbs_grouping) == "", "unknown_wbs", wbs_grouping),
    platform = if_else(is.na(platform) | trimws(platform) == "", "unknown_platform", platform),
    region = if_else(is.na(region) | trimws(region) == "", "unknown_region", region),
    country = if_else(is.na(country) | trimws(country) == "", "unknown_country", country),
    currency_cc = if_else(is.na(currency_cc) | trimws(currency_cc) == "", "unknown_currency", currency_cc)
  ) %>%
  filter(
    !is.na(value),
    value > 0,
    !is.na(wtg_num_new),
    wtg_num_new > 0,
    region %in% valid_region_levels,
    platform != "unknown_platform",
    country != "unknown_country",
    currency_cc != "unknown_currency"
  )

valid_scope_tbl <- xgb_results_region_platform_wbs_raw %>%
  filter(
    status == "ok",
    region %in% valid_region_levels,
    !is.na(platform),
    platform != "unknown_platform",
    !is.na(wbs_scope),
    n_train >= min_train_threshold
  ) %>%
  mutate(n_total = n_train + n_test) %>%
  filter(vapply(model, function(m) !is.null(xgb_models_region_platform_wbs[[m]]), logical(1)))

global_scope_tbl <- xgb_results_global_wbs_raw %>%
  filter(
    status == "ok",
    n_train >= min_train_threshold
  ) %>%
  mutate(n_total = n_train + n_test) %>%
  filter(vapply(model, function(m) !is.null(xgb_models_global_wbs[[m]]), logical(1)))

region_tbl <- df_model_base_dashboard %>%
  count(region, sort = TRUE) %>%
  filter(region %in% unique(c(valid_scope_tbl$region, df_model_base_dashboard$region))) %>%
  arrange(region)

region_choices_named <- c(
  "Select region" = "",
  stats::setNames(region_tbl$region, region_tbl$region)
)

choose_model_scope <- function(region_value, platform_value, wbs_value) {
  exact_row <- valid_scope_tbl %>%
    filter(
      region == region_value,
      platform == platform_value,
      wbs_scope == wbs_value
    ) %>%
    arrange(test_rmse) %>%
    dplyr::slice_head(n = 1)
  
  if (nrow(exact_row) > 0) {
    return(list(
      scope_type = "region_platform_wbs",
      scope_label = "Region + platform + WBS model",
      model_key = exact_row$model[[1]],
      metrics = exact_row
    ))
  }
  
  global_row <- global_scope_tbl %>%
    filter(wbs_scope == wbs_value) %>%
    arrange(test_rmse) %>%
    dplyr::slice_head(n = 1)
  
  if (nrow(global_row) > 0) {
    return(list(
      scope_type = "global_wbs",
      scope_label = "Global + WBS model",
      model_key = global_row$model[[1]],
      metrics = global_row
    ))
  }
  
  fallback_row <- global_scope_tbl %>%
    filter(wbs_scope == "all_wbs") %>%
    arrange(test_rmse) %>%
    dplyr::slice_head(n = 1)
  
  if (nrow(fallback_row) > 0) {
    return(list(
      scope_type = "global_wbs",
      scope_label = "Global all-WBS fallback model",
      model_key = fallback_row$model[[1]],
      metrics = fallback_row
    ))
  }
  
  stop("No model with 80 or more training observations was found for the current selection.", call. = FALSE)
}

build_scope_bundle <- function(region_value, platform_value, wbs_value, scope_type) {
  df_model <- df_model_base_dashboard
  
  if (scope_type == "region_platform_wbs") {
    df_model <- df_model %>%
      filter(region == region_value, platform == platform_value)
    
    if (wbs_value != "all_wbs") {
      df_model <- df_model %>% filter(wbs_grouping == wbs_value)
    }
    
    n_total <- nrow(df_model)
    n_train <- floor(0.7 * n_total)
    
    if (n_total < 5 || n_train < 3 || (n_total - n_train) < 1) {
      stop("Too few observations for the selected region-platform-WBS scope.", call. = FALSE)
    }
    
    train_raw <- df_model %>% dplyr::slice(1:n_train)
    test_raw <- df_model %>% dplyr::slice((n_train + 1):n_total)
    
    rec_model <- recipe(value ~ ., data = train_raw) %>%
      update_role(project_id, identifier, revision, sapp_proj_code, new_role = "id") %>%
      step_rm(
        region,
        platform,
        conversion_source,
        unitary_ammount_std_curr_cc,
        value_total_eur
      ) %>%
      step_impute_median(all_numeric_predictors()) %>%
      step_string2factor(all_nominal_predictors()) %>%
      step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
      step_novel(all_nominal_predictors(), new_level = "new") %>%
      step_zv(all_predictors()) %>%
      step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
      step_zv(all_predictors())
    
    rec_prep <- prep(rec_model, training = train_raw)
    
    return(list(
      train_raw = train_raw,
      test_raw = test_raw,
      rec_prep = rec_prep
    ))
  }
  
  if (scope_type == "global_wbs") {
    if (wbs_value != "all_wbs") {
      df_model <- df_model %>% filter(wbs_grouping == wbs_value)
    }
    
    n_total <- nrow(df_model)
    n_train <- floor(0.7 * n_total)
    
    if (n_total < 5 || n_train < 3 || (n_total - n_train) < 1) {
      stop("Too few observations for the selected global WBS scope.", call. = FALSE)
    }
    
    train_raw <- df_model %>% dplyr::slice(1:n_train)
    test_raw <- df_model %>% dplyr::slice((n_train + 1):n_total)
    
    rec_model <- recipe(value ~ ., data = train_raw) %>%
      update_role(project_id, identifier, revision, sapp_proj_code, new_role = "id") %>%
      step_rm(
        conversion_source,
        unitary_ammount_std_curr_cc,
        value_total_eur
      ) %>%
      step_impute_median(all_numeric_predictors()) %>%
      step_string2factor(all_nominal_predictors()) %>%
      step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
      step_novel(all_nominal_predictors(), new_level = "new") %>%
      step_zv(all_predictors()) %>%
      step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
      step_zv(all_predictors())
    
    rec_prep <- prep(rec_model, training = train_raw)
    
    return(list(
      train_raw = train_raw,
      test_raw = test_raw,
      rec_prep = rec_prep
    ))
  }
  
  stop("Unknown scope type.", call. = FALSE)
}

make_new_case <- function(region_value, country_value, platform_value, wbs_value, hub_height_value, wtg_num_value, currency_value) {
  tibble(
    project_id = "dashboard_case",
    identifier = "dashboard_case",
    revision = "dashboard",
    sapp_proj_code = "dashboard_case",
    wbs_grouping = wbs_value,
    platform = platform_value,
    hub_height = as.numeric(hub_height_value),
    wtg_num_new = as.numeric(wtg_num_value),
    region = region_value,
    country = country_value,
    currency_cc = currency_value,
    unitary_ammount_std_curr_cc = NA_real_,
    value_total_eur = NA_real_,
    value = NA_real_,
    conversion_source = "dashboard_input"
  )
}

get_model_object <- function(scope_type, model_key) {
  if (scope_type == "region_platform_wbs") {
    return(xgb_models_region_platform_wbs[[model_key]])
  }
  if (scope_type == "global_wbs") {
    return(xgb_models_global_wbs[[model_key]])
  }
  NULL
}

predict_dashboard_case <- function(region_value, country_value, platform_value, wbs_value, hub_height_value, wtg_num_value, currency_value) {
  tryCatch(
    {
      selected_model <- choose_model_scope(region_value, platform_value, wbs_value)
      scope_bundle <- build_scope_bundle(region_value, platform_value, wbs_value, selected_model$scope_type)
      model_object <- get_model_object(selected_model$scope_type, selected_model$model_key)
      
      if (is.null(model_object)) {
        stop("The selected trained model object was not found in memory.", call. = FALSE)
      }
      
      new_case <- make_new_case(
        region_value = region_value,
        country_value = country_value,
        platform_value = platform_value,
        wbs_value = wbs_value,
        hub_height_value = hub_height_value,
        wtg_num_value = wtg_num_value,
        currency_value = currency_value
      )
      
      baked_new <- bake(scope_bundle$rec_prep, new_data = new_case) %>%
        dplyr::select(value, where(is.numeric))
      
      baked_train <- bake(scope_bundle$rec_prep, new_data = scope_bundle$train_raw) %>%
        dplyr::select(value, where(is.numeric))
      
      baked_test <- bake(scope_bundle$rec_prep, new_data = scope_bundle$test_raw) %>%
        dplyr::select(value, where(is.numeric))
      
      baked_new_x <- baked_new %>% dplyr::select(-value) %>% as.matrix()
      train_x <- baked_train %>% dplyr::select(-value) %>% as.matrix()
      test_x <- baked_test %>% dplyr::select(-value) %>% as.matrix()
      
      pred_value <- as.numeric(predict(model_object, newdata = baked_new_x))
      pred_train <- as.numeric(predict(model_object, newdata = train_x))
      pred_test <- as.numeric(predict(model_object, newdata = test_x))
      
      if (length(pred_value) != 1 || is.na(pred_value)) {
        stop("Prediction could not be computed for the selected case.", call. = FALSE)
      }
      
      train_rmse_real <- rmse_vec_local(scope_bundle$train_raw$value, pred_train)
      test_rmse_real <- rmse_vec_local(scope_bundle$test_raw$value, pred_test)
      
      naive_value <- mean(scope_bundle$train_raw$value, na.rm = TRUE)
      naive_test_rmse <- rmse_vec_local(scope_bundle$test_raw$value, rep(naive_value, nrow(scope_bundle$test_raw)))
      improvement_pct <- 100 * (naive_test_rmse - test_rmse_real) / naive_test_rmse
      gap_pct <- 100 * abs(test_rmse_real - train_rmse_real) / train_rmse_real
      
      vip_tbl <- tryCatch(
        {
          xgboost::xgb.importance(model = model_object) %>%
            as_tibble() %>%
            transmute(Feature = Feature, Importance = Gain)
        },
        error = function(e) tibble(Feature = "Importance not available", Importance = 0)
      )
      
      warning_lines <- c()
      
      if (selected_model$scope_type != "region_platform_wbs") {
        warning_lines <- c(
          warning_lines,
          paste0("Selected scope: ", selected_model$scope_label, ". A more granular model was not available for this exact combination.")
        )
      } else {
        warning_lines <- c(
          warning_lines,
          paste0("Selected scope: ", selected_model$scope_label, ".")
        )
      }
      
      if (gap_pct > 35) {
        warning_lines <- c(warning_lines, "Caution: this model shows a relatively high train-test gap.")
      }
      
      if ((test_rmse_real / pred_value) > 0.35) {
        warning_lines <- c(warning_lines, "Caution: the expected uncertainty is relatively high relative to the point estimate.")
      }
      
      list(
        error = NULL,
        prediction = pred_value,
        train_rmse = train_rmse_real,
        test_rmse = test_rmse_real,
        naive_test_rmse = naive_test_rmse,
        improvement_pct = improvement_pct,
        gap_pct = gap_pct,
        n_train = selected_model$metrics$n_train[[1]],
        n_test = selected_model$metrics$n_test[[1]],
        n_total = selected_model$metrics$n_total[[1]],
        model_key = selected_model$model_key,
        model_scope = selected_model$scope_label,
        vip_table = vip_tbl,
        train_raw = scope_bundle$train_raw,
        warnings = paste(warning_lines, collapse = "\n"),
        user_region = region_value,
        user_country = country_value,
        user_platform = platform_value,
        user_wbs = wbs_value,
        user_hub_height = hub_height_value,
        user_currency = currency_value,
        user_wtg_num = wtg_num_value
      )
    },
    error = function(e) {
      list(error = conditionMessage(e))
    }
  )
}

find_similar_projects <- function(reference_df, region_value, country_value, platform_value, wbs_value, hub_height_value, wtg_num_value, currency_value) {
  tryCatch(
    {
      ref_df <- reference_df
      
      ref_region <- ref_df %>% filter(region == region_value)
      if (nrow(ref_region) > 0) ref_df <- ref_region
      
      ref_country <- ref_df %>% filter(country == country_value)
      if (nrow(ref_country) > 0) ref_df <- ref_country
      
      ref_platform <- ref_df %>% filter(platform == platform_value)
      if (nrow(ref_platform) > 0) ref_df <- ref_platform
      
      if (wbs_value != "all_wbs") {
        ref_wbs <- ref_df %>% filter(wbs_grouping == wbs_value)
        if (nrow(ref_wbs) > 0) ref_df <- ref_wbs
      }
      
      ref_currency <- ref_df %>% filter(currency_cc == currency_value)
      if (nrow(ref_currency) > 0) ref_df <- ref_currency
      
      ref_df %>%
        mutate(
          hub_height_distance = abs(hub_height - as.numeric(hub_height_value)),
          wtg_distance = abs(wtg_num_new - as.numeric(wtg_num_value))
        ) %>%
        arrange(hub_height_distance, wtg_distance, desc(value)) %>%
        select(
          identifier,
          region,
          country,
          platform,
          wbs_grouping,
          currency_cc,
          hub_height,
          wtg_num_new,
          value
        ) %>%
        dplyr::slice_head(n = 100)
    },
    error = function(e) {
      tibble(identifier = paste("Error:", conditionMessage(e)))
    }
  )
}

driver_plot <- function(vip_table) {
  vip_table %>%
    dplyr::slice_head(n = 10) %>%
    mutate(Feature = fct_reorder(Feature, Importance)) %>%
    ggplot(aes(x = Feature, y = Importance)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(
      title = "Key drivers behind the selected estimate",
      x = NULL,
      y = "Gain"
    ) +
    theme_classic(base_size = 14)
}

portfolio_heatmap_data <- function() {
  plot_df <- valid_scope_tbl %>%
    group_by(region, platform) %>%
    mutate(region_platform_rank = median(test_rmse, na.rm = TRUE)) %>%
    ungroup() %>%
    group_by(wbs_scope) %>%
    mutate(wbs_rank = median(test_rmse, na.rm = TRUE)) %>%
    ungroup() %>%
    arrange(region, region_platform_rank, wbs_scope) %>%
    mutate(
      region_platform = paste(region, platform, sep = "\n"),
      wbs_scope = factor(wbs_scope, levels = rev(unique(wbs_scope[order(wbs_rank)])))
    )
  
  x_axis_tbl <- plot_df %>%
    distinct(region, platform, region_platform, region_platform_rank) %>%
    arrange(region, region_platform_rank) %>%
    mutate(x_num = row_number())
  
  plot_df %>%
    left_join(
      x_axis_tbl %>% select(region, platform, x_num, region_platform),
      by = c("region", "platform", "region_platform")
    ) %>%
    mutate(y_num = as.numeric(wbs_scope))
}

portfolio_heatmap_plot <- function(plot_df) {
  x_lab_df <- plot_df %>% distinct(x_num, region_platform) %>% arrange(x_num)
  
  ggplot(plot_df, aes(x = x_num, y = y_num, fill = test_rmse)) +
    geom_tile(color = "white", linewidth = 0.7) +
    geom_text(
      aes(y = y_num + 0.16, label = fmt_num(round(test_rmse))),
      size = 3.0,
      fontface = "bold",
      color = "black"
    ) +
    scale_x_continuous(
      breaks = x_lab_df$x_num,
      labels = x_lab_df$region_platform,
      expand = expansion(add = 0.4)
    ) +
    scale_y_continuous(
      breaks = seq_along(levels(plot_df$wbs_scope)),
      labels = levels(plot_df$wbs_scope),
      expand = expansion(add = 0.5)
    ) +
    scale_fill_gradientn(
      colours = c("#f2f0f7", "#cbc9e2", "#9e9ac8", "#6a51a3"),
      trans = "log10",
      labels = fmt_num
    ) +
    labs(
      title = "Validated model performance by region, platform and WBS",
      subtitle = "Only combinations with 80 or more training observations are shown",
      x = "Region and platform",
      y = "WBS grouping",
      fill = "Test RMSE"
    ) +
    theme_classic(base_size = 14) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      plot.title = element_text(face = "bold")
    )
}

empty_text <- "Select Region, Country, Platform, WBS grouping, Hub height, Number of turbines, and Currency."

ui <- fluidPage(
  theme = bs_theme(version = 5, bootswatch = "flatly"),
  tags$head(
    tags$style(HTML("
      .main-panel-fill { min-height: 78vh; }
      .dt-fill { height: 70vh; overflow-y: auto; }
    "))
  ),
  titlePanel("Preliminary Value Decision Support Dashboard"),
  sidebarLayout(
    sidebarPanel(
      width = 3,
      selectInput(
        "region",
        "Region",
        choices = region_choices_named,
        selected = "",
        selectize = FALSE
      ),
      selectInput(
        "country",
        "Country",
        choices = c("Select country" = ""),
        selected = "",
        selectize = FALSE
      ),
      selectInput(
        "platform",
        "Platform",
        choices = c("Select platform" = ""),
        selected = "",
        selectize = FALSE
      ),
      selectInput(
        "wbs_scope",
        "WBS grouping",
        choices = c("Select WBS grouping" = ""),
        selected = "",
        selectize = FALSE
      ),
      numericInput(
        "hub_height",
        "Hub height",
        value = 100,
        min = 0,
        step = 0.5
      ),
      numericInput(
        "wtg_num_new",
        "Number of turbines",
        value = 1,
        min = 1,
        step = 1
      ),
      selectInput(
        "currency_cc",
        "Currency",
        choices = c("Select currency" = ""),
        selected = "",
        selectize = FALSE
      ),
      actionButton("run_pred", "Run estimate", class = "btn-primary"),
      hr(),
      p("The dashboard provides a point estimate, key drivers, model reliability information, and contextual project examples. The benchmark shown in the model card is the matched naive baseline using the mean value from the selected training scope.")
    ),
    mainPanel(
      width = 9,
      class = "main-panel-fill",
      fluidRow(
        column(
          3,
          wellPanel(
            h4("Predicted value per turbine"),
            textOutput("pred_value"),
            tags$small("EUR per turbine")
          )
        ),
        column(
          3,
          wellPanel(
            h4("Model scope used"),
            textOutput("scope_value"),
            tags$small("Best available scope")
          )
        ),
        column(
          3,
          wellPanel(
            h4("Training observations"),
            textOutput("ntrain_value"),
            tags$small("Train rows in selected model")
          )
        ),
        column(
          3,
          wellPanel(
            h4("Test observations"),
            textOutput("ntest_value"),
            tags$small("Test rows in selected model")
          )
        )
      ),
      fluidRow(
        column(
          12,
          wellPanel(
            h4("Reliability and caution flags"),
            verbatimTextOutput("warning_text")
          )
        )
      ),
      tabsetPanel(
        tabPanel("Key drivers", plotOutput("driver_plot", height = "78vh")),
        tabPanel("Similar historical projects", div(class = "dt-fill", DTOutput("similar_tbl"))),
        tabPanel(
          "Model card",
          div(
            class = "dt-fill",
            h4("Model performance summary"),
            DTOutput("model_card_tbl"),
            br(),
            h4("Inputs used in current prediction"),
            DTOutput("inputs_used_tbl")
          )
        ),
        tabPanel("Portfolio heatmap", plotOutput("portfolio_plot", height = "78vh"))
      )
    )
  )
)

server <- function(input, output, session) {
  observeEvent(input$region, {
    freezeReactiveValue(input, "country")
    freezeReactiveValue(input, "platform")
    freezeReactiveValue(input, "currency_cc")
    freezeReactiveValue(input, "wbs_scope")
    
    if (is.null(input$region) || input$region == "") {
      updateSelectInput(session, "country", choices = c("Select country" = ""), selected = "")
      updateSelectInput(session, "platform", choices = c("Select platform" = ""), selected = "")
      updateSelectInput(session, "currency_cc", choices = c("Select currency" = ""), selected = "")
      updateSelectInput(session, "wbs_scope", choices = c("Select WBS grouping" = ""), selected = "")
      return()
    }
    
    region_df <- df_model_base_dashboard %>%
      filter(region == input$region)
    
    country_tbl <- region_df %>%
      count(country, sort = TRUE) %>%
      filter(!is.na(country), country != "unknown_country")
    
    country_choices <- stats::setNames(country_tbl$country, country_tbl$country)
    
    updateSelectInput(session, "country", choices = c("Select country" = "", country_choices), selected = "")
    updateSelectInput(session, "platform", choices = c("Select platform" = ""), selected = "")
    updateSelectInput(session, "currency_cc", choices = c("Select currency" = ""), selected = "")
    updateSelectInput(session, "wbs_scope", choices = c("Select WBS grouping" = ""), selected = "")
  }, ignoreInit = FALSE)
  
  observeEvent(c(input$region, input$country), {
    freezeReactiveValue(input, "platform")
    freezeReactiveValue(input, "currency_cc")
    freezeReactiveValue(input, "wbs_scope")
    
    if (is.null(input$region) || input$region == "") {
      updateSelectInput(session, "platform", choices = c("Select platform" = ""), selected = "")
      updateSelectInput(session, "currency_cc", choices = c("Select currency" = ""), selected = "")
      updateSelectInput(session, "wbs_scope", choices = c("Select WBS grouping" = ""), selected = "")
      return()
    }
    
    filtered_df <- df_model_base_dashboard %>%
      filter(region == input$region)
    
    if (!is.null(input$country) && input$country != "") {
      filtered_df <- filtered_df %>% filter(country == input$country)
    }
    
    platform_tbl <- filtered_df %>%
      count(platform, sort = TRUE) %>%
      filter(!is.na(platform), platform != "unknown_platform")
    
    platform_choices <- stats::setNames(platform_tbl$platform, platform_tbl$platform)
    
    currency_tbl <- filtered_df %>%
      count(currency_cc, sort = TRUE) %>%
      filter(!is.na(currency_cc), currency_cc != "unknown_currency")
    
    currency_choices <- stats::setNames(currency_tbl$currency_cc, currency_tbl$currency_cc)
    
    updateSelectInput(session, "platform", choices = c("Select platform" = "", platform_choices), selected = "")
    updateSelectInput(session, "currency_cc", choices = c("Select currency" = "", currency_choices), selected = "")
    updateSelectInput(session, "wbs_scope", choices = c("Select WBS grouping" = ""), selected = "")
  }, ignoreInit = FALSE)
  
  observeEvent(c(input$region, input$platform), {
    freezeReactiveValue(input, "wbs_scope")
    
    if (is.null(input$region) || input$region == "" || is.null(input$platform) || input$platform == "") {
      updateSelectInput(session, "wbs_scope", choices = c("Select WBS grouping" = ""), selected = "")
      return()
    }
    
    exact_wbs_tbl <- valid_scope_tbl %>%
      filter(region == input$region, platform == input$platform) %>%
      group_by(wbs_scope) %>%
      summarise(
        best_test_rmse = min(test_rmse, na.rm = TRUE),
        n_train = max(n_train, na.rm = TRUE),
        n_test = max(n_test, na.rm = TRUE),
        n_total = max(n_total, na.rm = TRUE),
        availability = "Exact",
        .groups = "drop"
      )
    
    global_wbs_tbl <- global_scope_tbl %>%
      group_by(wbs_scope) %>%
      summarise(
        best_test_rmse = min(test_rmse, na.rm = TRUE),
        n_train = max(n_train, na.rm = TRUE),
        n_test = max(n_test, na.rm = TRUE),
        n_total = max(n_total, na.rm = TRUE),
        availability = "Fallback",
        .groups = "drop"
      )
    
    wbs_tbl <- bind_rows(exact_wbs_tbl, global_wbs_tbl) %>%
      group_by(wbs_scope) %>%
      summarise(
        best_test_rmse = min(best_test_rmse, na.rm = TRUE),
        n_train = max(n_train, na.rm = TRUE),
        n_test = max(n_test, na.rm = TRUE),
        n_total = max(n_total, na.rm = TRUE),
        availability = if_else(any(availability == "Exact"), "Exact", "Fallback"),
        .groups = "drop"
      ) %>%
      arrange(desc(wbs_scope == "all_wbs"), desc(availability == "Exact"), best_test_rmse, desc(n_total), wbs_scope)
    
    if (nrow(wbs_tbl) == 0) {
      updateSelectInput(session, "wbs_scope", choices = c("Select WBS grouping" = ""), selected = "")
      return()
    }
    
    wbs_labels <- paste0(wbs_tbl$wbs_scope, " (", wbs_tbl$availability, ")")
    wbs_choices <- stats::setNames(wbs_tbl$wbs_scope, wbs_labels)
    
    updateSelectInput(
      session,
      "wbs_scope",
      choices = c("Select WBS grouping" = "", wbs_choices),
      selected = ""
    )
  }, ignoreInit = FALSE)
  
  prediction_result <- eventReactive(input$run_pred, {
    if (input$region == "" || input$country == "" || input$platform == "" || input$wbs_scope == "" || input$currency_cc == "") {
      return(list(error = "Please complete all dropdown selections before running the estimate."))
    }
    
    req(!is.na(input$hub_height))
    req(!is.na(input$wtg_num_new))
    
    predict_dashboard_case(
      region_value = input$region,
      country_value = input$country,
      platform_value = input$platform,
      wbs_value = input$wbs_scope,
      hub_height_value = input$hub_height,
      wtg_num_value = input$wtg_num_new,
      currency_value = input$currency_cc
    )
  })
  
  similar_tbl_reactive <- eventReactive(input$run_pred, {
    req(prediction_result())
    
    if (!is.null(prediction_result()$error)) {
      return(tibble(identifier = prediction_result()$error))
    }
    
    find_similar_projects(
      reference_df = prediction_result()$train_raw,
      region_value = input$region,
      country_value = input$country,
      platform_value = input$platform,
      wbs_value = input$wbs_scope,
      hub_height_value = input$hub_height,
      wtg_num_value = input$wtg_num_new,
      currency_value = input$currency_cc
    )
  })
  
  output$pred_value <- renderText({
    if (is.null(input$region) || input$region == "") return(empty_text)
    if (input$run_pred < 1) return(empty_text)
    req(prediction_result())
    if (!is.null(prediction_result()$error)) return(prediction_result()$error)
    paste0("€ ", fmt_num(round(prediction_result()$prediction)))
  })
  
  output$scope_value <- renderText({
    if (is.null(input$region) || input$region == "") return(empty_text)
    if (input$run_pred < 1) return(empty_text)
    req(prediction_result())
    if (!is.null(prediction_result()$error)) return(prediction_result()$error)
    prediction_result()$model_scope
  })
  
  output$ntrain_value <- renderText({
    if (is.null(input$region) || input$region == "") return(empty_text)
    if (input$run_pred < 1) return(empty_text)
    req(prediction_result())
    if (!is.null(prediction_result()$error)) return(prediction_result()$error)
    as.character(prediction_result()$n_train)
  })
  
  output$ntest_value <- renderText({
    if (is.null(input$region) || input$region == "") return(empty_text)
    if (input$run_pred < 1) return(empty_text)
    req(prediction_result())
    if (!is.null(prediction_result()$error)) return(prediction_result()$error)
    as.character(prediction_result()$n_test)
  })
  
  output$warning_text <- renderText({
    if (is.null(input$region) || input$region == "") return(empty_text)
    if (input$run_pred < 1) return(empty_text)
    req(prediction_result())
    if (!is.null(prediction_result()$error)) return(prediction_result()$error)
    prediction_result()$warnings
  })
  
  output$driver_plot <- renderPlot({
    validate(need(input$run_pred >= 1, empty_text))
    req(prediction_result())
    validate(need(is.null(prediction_result()$error), prediction_result()$error))
    print(driver_plot(prediction_result()$vip_table))
  })
  
  output$similar_tbl <- renderDT({
    if (input$run_pred < 1) {
      return(
        datatable(
          tibble(Message = empty_text),
          rownames = FALSE,
          options = list(dom = "t")
        )
      )
    }
    
    req(similar_tbl_reactive())
    datatable(
      similar_tbl_reactive(),
      rownames = FALSE,
      options = list(
        pageLength = 100,
        lengthMenu = c(10, 25, 50, 100),
        scrollX = TRUE,
        scrollY = "65vh"
      )
    )
  })
  
  output$model_card_tbl <- renderDT({
    if (input$run_pred < 1) {
      return(
        datatable(
          tibble(Message = empty_text),
          rownames = FALSE,
          options = list(dom = "t")
        )
      )
    }
    
    req(prediction_result())
    
    if (!is.null(prediction_result()$error)) {
      return(
        datatable(
          tibble(Message = prediction_result()$error),
          rownames = FALSE,
          options = list(dom = "t")
        )
      )
    }
    
    card_tbl <- tibble(
      Metric = c(
        "Model key",
        "Model scope",
        "Training observations",
        "Test observations",
        "Total observations",
        "Train RMSE",
        "Test RMSE",
        "Matched naive baseline",
        "Matched naive test RMSE",
        "Improvement vs matched naive",
        "Train-test gap"
      ),
      Value = c(
        prediction_result()$model_key,
        prediction_result()$model_scope,
        prediction_result()$n_train,
        prediction_result()$n_test,
        prediction_result()$n_total,
        fmt_num(round(prediction_result()$train_rmse)),
        fmt_num(round(prediction_result()$test_rmse)),
        "Mean value of the selected training scope",
        fmt_num(round(prediction_result()$naive_test_rmse)),
        paste0(round(prediction_result()$improvement_pct, 1), "%"),
        paste0(round(prediction_result()$gap_pct, 1), "%")
      )
    )
    
    datatable(
      card_tbl,
      rownames = FALSE,
      options = list(dom = "t", pageLength = nrow(card_tbl), scrollX = TRUE)
    )
  })
  
  output$inputs_used_tbl <- renderDT({
    if (input$run_pred < 1) {
      return(
        datatable(
          tibble(Message = empty_text),
          rownames = FALSE,
          options = list(dom = "t")
        )
      )
    }
    
    req(prediction_result())
    
    if (!is.null(prediction_result()$error)) {
      return(
        datatable(
          tibble(Message = prediction_result()$error),
          rownames = FALSE,
          options = list(dom = "t")
        )
      )
    }
    
    inputs_tbl <- tibble(
      Input = c(
        "Region",
        "Country",
        "Platform",
        "WBS grouping",
        "Hub height",
        "Number of turbines",
        "Currency"
      ),
      Value = c(
        prediction_result()$user_region,
        prediction_result()$user_country,
        prediction_result()$user_platform,
        prediction_result()$user_wbs,
        prediction_result()$user_hub_height,
        prediction_result()$user_wtg_num,
        prediction_result()$user_currency
      ),
      Source = c(
        "User-selected",
        "User-selected",
        "User-selected",
        "User-selected",
        "User-selected",
        "User-selected",
        "User-selected"
      )
    )
    
    datatable(
      inputs_tbl,
      rownames = FALSE,
      options = list(dom = "t", pageLength = nrow(inputs_tbl), scrollX = TRUE)
    )
  })
  
  output$portfolio_plot <- renderPlot({
    print(portfolio_heatmap_plot(portfolio_heatmap_data()))
  })
}

shinyApp(ui = ui, server = server)