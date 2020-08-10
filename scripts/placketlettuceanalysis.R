library(PlackettLuce)
library(tidyr)
library(dplyr)

groups <- survey %>%
  group_by(algo1, algo2, choosen_algo) %>%
  count(choosen_algo)

groups_wins1 <- groups %>%
  filter(choosen_algo == 1) %>%
  rename(w_ij = n) %>%
  ungroup(choosen_algo) %>%
  select(-choosen_algo)


groups_wins2 <- groups %>%
  filter(choosen_algo == 2) %>%
  rename(w_ji = n) %>%
  ungroup(choosen_algo) %>%
  select(-choosen_algo)


groups_tied <- groups %>%
  filter(choosen_algo == 0) %>%
  rename(t_ij = n) %>%
  ungroup(choosen_algo) %>%
  select(-choosen_algo)

tests_done <- survey %>%
  group_by(algo1, algo2) %>%
  count()

groups_joined <- merge(groups_wins1, groups_wins2, all = TRUE)
groups_joined <- merge(groups_joined, groups_tied, all = TRUE)
groups_joined[is.na(groups_joined)] <- 0

i_wins <- data.frame(Winner = groups_joined$algo1, Loser = groups_joined$algo2, stringsAsFactors=FALSE)
j_wins <- data.frame(Winner = groups_joined$algo2, Loser = groups_joined$algo1, stringsAsFactors=FALSE)
ties <- data.frame(Winner = array(split(groups_joined[c("algo1", "algo2")], 1:nrow(groups_joined)), nrow(groups_joined)),
                   Loser = rep(NA, nrow(groups_joined)), stringsAsFactors=FALSE)

R <- as.rankings(rbind(i_wins, j_wins, ties), input = "orderings")
w <- unlist(groups_joined[c("w_ij", "w_ji", "t_ij")])
mod <- PlackettLuce(R, weights = w + .Machine$double.xmin, npseudo = 0)
qv <- qvcalc(mod)
plot(qv, xlab = "Brand of pudding", ylab = "Worth (log)", xatx= 'n', main = NULL)
axis(1, labels = rownames(qv[[1]]), at=1:10, las=2)

