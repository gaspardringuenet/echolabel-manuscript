library(tidyverse)
library(ggpmisc)
library(glue)

layer <- read_csv(
  "~/Projects/_manuscripts/echolabel-manuscript/data/layer.csv", 
  col_types = cols(time = col_datetime(format = "%Y-%m-%d %H:%M:%S"))
)


t0 <- strptime("2017-04-26 10:00:00", format="%Y-%m-%d %H:%M:%S", tz="UTC")
t0 <- as.POSIXct(t0)

ggplot(layer, aes(x=time, y=depth, color=delta_Sv)) +
  geom_point() +
  geom_vline(xintercept=t0) + 
  scale_color_distiller(palette = "Spectral") +
  scale_y_reverse() +
  theme_light() +
  labs(
    x = "Ping Time",
    y = "Depth [m]",
    color = "ΔMVBS 70-38 kHz [dB]",
    title = "Layer average depth profile - A link between peaks and ΔMVBS ?",
    subtitle = "Depth is averaged, weighted by 38 kHz NASC, and smoothed with rolling mean",
    caption = "Data from ABRAÇOS II cruise"
  )


# Part I. - Global correlation between speed and delta_Sv ---------------------

# Simple linear correlation

layer <- layer |> 
  filter(time > t0)

correlation <- round(cor(layer$speed_smooth, layer$delta_Sv, method = "pearson"), 2)

layer |> 
  ggplot(aes(x=speed_smooth, y=delta_Sv)) +
  geom_point(alpha=0.5) +
  stat_poly_line(color="red") +
  stat_poly_eq(use_label(c("eq", "R2", "P"))) +
  scale_color_viridis_c() +
  labs(
    x = "Vertical speed [m.s-1]",
    y = "ΔMVBS 70-38 kHz [dB]",
    title = glue("Correlation between vertical speed and ΔMVBS (Pearson: {correlation})"),
    subtitle = "Vertical speed is derived from smoothed average depth",
    caption = "Data from ABRAÇOS II cruise"
  ) +
  theme_light()

model <- lm(delta_Sv ~ speed_smooth, layer)
summary(model)
par(mfrow=c(2, 2))
plot(model)
par(mar = c(0, 0, 0, 0))


max(cooks.distance(model))

# Sign correlation
# Are delta_Sv means different between the -/+ vertical speed groups?

layer <- layer |> 
  mutate(swimming_up = speed_smooth < 0)

mean_delta_Sv_up <- layer |> 
  group_by(swimming_up) |> 
  summarize(
    mean = mean(delta_Sv),
    sd = sd(delta_Sv),
    n = n()
  )
 
ggplot(layer, aes(x=delta_Sv, fill=swimming_up)) +
  geom_histogram(alpha=0.7, bins=50) +
  geom_vline(data=mean_delta_Sv_up, aes(xintercept=mean, color=swimming_up)) +
  theme_light()

layer |> 
  ggplot(aes(x=speed_smooth, y=delta_Sv, color=swimming_up)) +
  geom_point() +
  stat_poly_line() +
  stat_poly_eq(use_label(c("eq", "R2", "P"))) +
  labs(
    x = "Vertical speed [m.s-1]",
    y = "ΔMVBS 70-38 kHz [dB]",
    color = "Moving Up",
    caption = "Data from ABRAÇOS II cruise"
  ) +
  theme_light()

model <- lm(delta_Sv ~ speed_smooth * swimming_up, layer)
summary(model)
par(mfrow=c(2, 2))
plot(model)
par(mar = c(0, 0, 0, 0))


layer |> 
  ggplot(aes(x=speed_smooth, y=delta_Sv)) +
  geom_point(alpha=0.5) +
  stat_poly_line(formula = y ~ poly(x, 2, raw = TRUE), color="red") +
  stat_poly_eq(formula = y ~ poly(x, 2, raw = TRUE), use_label("eq")) +
  scale_color_viridis_c() +
  labs(
    x = "Vertical speed [m.s-1]",
    y = "ΔMVBS 70-38 kHz [dB]",
    caption = "Data from ABRAÇOS II cruise"
  ) +
  theme_light()


layer$speed_squared = layer$speed_smooth ** 2
model <- lm(delta_Sv ~ speed_smooth + speed_squared, layer)
summary(model)
par(mfrow=c(2, 2))
plot(model)
par(mar = c(0, 0, 0, 0))
