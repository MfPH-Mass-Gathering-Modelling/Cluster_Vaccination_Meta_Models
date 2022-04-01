"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description:
    Contains functions for downloading vaccination data from Canadian provinces and returning a dataframe.
"""
import pandas as pd

pd.options.mode.chained_assignment = None  # We don't want the warnings, default='warn'


def ontario():
    """Download and clean the data on Ontario's Vaccination program.
    This data is housed at https://data.ontario.ca/en/dataset/covid-19-vaccine-data-in-ontario.

    Returns:
        pandas.DataFrame: Cleaned data on Ontario's vaccination program.
    """
    # To access the data through pandas.
    vaccination_df = pd.read_csv(
        'https://data.ontario.ca/dataset/752ce2b7-c15a-4965-a3dc-397bf405e7cc/resource/8a89caa9-511c-4568-af89-7f2174b4378c/download/vaccine_doses.csv')
    # Convert report date to date type.
    # This makes things easier in terms of selecting between dates, first date (min) or laste date (max).
    vaccination_df.report_date = pd.to_datetime(vaccination_df.report_date)

    # Select interested fields
    selection = [
        'report_date',
        'previous_day_at_least_one',
        'previous_day_fully_vaccinated',
        'previous_day_3doses',
        'previous_day_total_doses_administered',
        'total_individuals_at_least_one',
        'total_individuals_partially_vaccinated',
        'total_individuals_fully_vaccinated',
        'total_individuals_3doses',
        'total_doses_administered'
    ]
    vaccination_df_select = vaccination_df[selection]
    # looking at the data in excel there seems to be a change in how the data is recorded.  That date is.
    date_of_change = vaccination_df_select.report_date[
        vaccination_df_select.total_individuals_partially_vaccinated.notnull()].min()

    # for all dates before date_of_change we need to know the 'total_individuals_partially_vaccinated'.
    null_rows = vaccination_df_select.total_individuals_partially_vaccinated.isnull()
    vaccination_df_select.total_individuals_partially_vaccinated[null_rows] = (
                vaccination_df_select.total_individuals_at_least_one[null_rows] -
                vaccination_df_select.total_individuals_fully_vaccinated[null_rows])
    vaccination_df_select.head(40)

    # checking booster shot trecords around time of change in record keeping
    start_date = '2021-11-25'
    end_date = '2021-12-10'
    date_selection = (vaccination_df_select.report_date >= start_date) & (vaccination_df_select.report_date <= end_date)
    vac_rec_change_df = vaccination_df_select.loc[date_selection]

    vac_rec_change_df['sum_check'] = (vac_rec_change_df.total_doses_administered -
                                      (vac_rec_change_df.total_individuals_at_least_one +
                                       vac_rec_change_df.total_individuals_fully_vaccinated +
                                       vac_rec_change_df.total_individuals_3doses.fillna(0)
                                       )
                                      )

    # So it seems that booster doses were not recorded until 3rd Dec 2021.
    #
    # But on that date were have 13855 doses come from?
    #
    # Aggh 3rd December they start recording Jansen jabs in total_individuals_fully_vaccinated. As put in the assoiciated data dictionary (https://data.ontario.ca/dataset/752ce2b7-c15a-4965-a3dc-397bf405e7cc/resource/29d182db-69c5-4cfb-894e-35227013689d/download/vaccine_open_data_dictionary_en_fr.xlsx first sheet):
    # "Prior to December 3, 2021, individuals who received one dose of Janssen (Johnson & Johnson) were not captured in this field. "
    #
    # Also worth noting from data dictionary for partially vaccinated non-Canadian approved vaccines are counted.

    # ## Adding missing total_individuals_at_least_one data, previous_day_at_least_one and total_individuals_partially_vaccinated
    #
    # We will assuming a contstant rate from vaccination program beginning to first record. The vaccination porgram began 15/12/2020 https://news.ontario.ca/en/release/59607/ontario-begins-rollout-of-covid-19-vaccine.

    # Will have to extend the vaccination_df_select dataframe back to 15/12/2020

    earlier_df = pd.DataFrame(pd.date_range(start='15/12/2020', end=vaccination_df_select.report_date[0]),
                              columns=['report_date'])
    vaccination_df_select = pd.merge(earlier_df, vaccination_df_select, on='report_date', how='outer')

    first_day_rec_at_least_one_val = vaccination_df_select.total_individuals_at_least_one[
        vaccination_df_select.total_individuals_at_least_one.notnull()].iloc[0]
    first_day_rec_at_least_one_date = \
    vaccination_df_select.report_date[vaccination_df_select.total_individuals_at_least_one.notnull()].iloc[0]
    days_from_15th = vaccination_df_select[vaccination_df_select.total_individuals_at_least_one.notnull()].index[0]

    mean_part_vac = first_day_rec_at_least_one_val / days_from_15th
    est_vals = [0]
    for i in range(1, days_from_15th):
        est_vals.append(est_vals[i - 1] + mean_part_vac)

    vaccination_df_select.total_individuals_at_least_one[
        vaccination_df_select.total_individuals_at_least_one.isnull()] = est_vals
    vaccination_df_select.previous_day_at_least_one[0] = 0
    vaccination_df_select.previous_day_at_least_one[
        vaccination_df_select.previous_day_at_least_one.isnull()] = mean_part_vac

    vaccination_df_select.total_individuals_partially_vaccinated[
        vaccination_df_select.total_individuals_partially_vaccinated.isnull()] = est_vals

    # ## Adding Unvaccinated column
    #
    # Ontario's population reached 14,755,211 on January 1, 2021 https://www.ontario.ca/page/ontario-demographic-quarterly-highlights-fourth-quarter-2020#:~:text=Ontario's%20population%20reached%2014%2C755%2C211%20on,quarter%20of%20the%20previous%20year..
    #
    # Note that a dates data is the vaccination totals recorded by 8 pm the previous day.

    ont_pop = 14755211
    vaccination_df_select['unvaccinated'] = ont_pop - vaccination_df_select.total_individuals_at_least_one
    vaccination_df_select

    # ## Fully vaccinated missing data.
    #
    # Asume no 2nd doses given until date of first recording
    vaccination_df_select.previous_day_fully_vaccinated[
        vaccination_df_select.previous_day_fully_vaccinated.isnull()] = 0
    req_index = vaccination_df_select.index[vaccination_df_select.total_individuals_fully_vaccinated.notnull()][0]
    vaccination_df_select.previous_day_fully_vaccinated[req_index] = \
    vaccination_df_select.total_individuals_fully_vaccinated[req_index]
    vaccination_df_select.total_individuals_fully_vaccinated[
        vaccination_df_select.total_individuals_fully_vaccinated.isnull()] = 0

    # ## Booster vaccination missing data.
    #
    # Booster program began 14th September 2021 https://news.ontario.ca/en/backgrounder/1000805/expanded-eligibility-for-third-doses-of-the-covid-19-vaccine.

    first_rec_booster_index = vaccination_df_select[vaccination_df_select.total_individuals_3doses.notnull()].index[0]
    first_rec_booster_val = vaccination_df_select.total_individuals_3doses[first_rec_booster_index]
    first_rec_booster_date = vaccination_df_select.report_date[first_rec_booster_index]

    booster_start_day = vaccination_df_select.index[vaccination_df_select.report_date == '2021-09-14'][0]
    days_missing_data = first_rec_booster_index - booster_start_day
    mean_booster_vac = first_rec_booster_val / days_missing_data
    est_vals = [0]
    for i in range(1, days_missing_data):
        est_vals.append(est_vals[i - 1] + mean_booster_vac)

    est_vals[-1] + mean_booster_vac

    vaccination_df_select.total_individuals_3doses[booster_start_day:booster_start_day + len(est_vals)] = est_vals
    vaccination_df_select.total_individuals_3doses.fillna(0, inplace=True)
    vaccination_df_select.iloc[booster_start_day - 1:first_rec_booster_index + 5]

    # previous_day_3doses has quie a few missing rows. Can quite simply calculate this using pandas .population_transitioning function.
    # Replace all na values with 0
    vaccination_df_select.previous_day_3doses = vaccination_df_select.total_individuals_3doses.diff()
    vaccination_df_select.previous_day_3doses.fillna(0, inplace=True)
    vaccination_df_select.iloc[booster_start_day - 1:first_rec_booster_index + 3]

    # ## Relabelling total_individuals_fully_vaccinated as total_at_least_two and creating a new total_individuals_fully_vaccinated.
    #
    # The data dictionary reveals the definition of total_individuals_fully_vaccinated is:
    # Cumulative number of individuals who are fully vaccinated. Fully vaccinated is defined as recieving:
    # * one dose of Janssen (Johnson & Johnson), or
    # * two doses of any Health Canada approved vaccine, or
    # * one dose of a non-Health Canada approved vaccine, followed by one dose of a Health Canada-approved vaccine,
    # or
    # * three doses of any Health Canada approved vaccine, or
    # * three doses of a vaccine, whether it is Health Canada approved or not.
    #
    # https://data.ontario.ca/dataset/752ce2b7-c15a-4965-a3dc-397bf405e7cc/resource/29d182db-69c5-4cfb-894e-35227013689d/download/vaccine_open_data_dictionary_en_fr.xlsx
    #
    # Therefore we need to do the following:

    vaccination_df_select['total_at_least_two'] = vaccination_df_select.total_individuals_fully_vaccinated
    vaccination_df_select.total_individuals_fully_vaccinated = vaccination_df_select.total_at_least_two - vaccination_df_select.total_individuals_3doses

    return vaccination_df_select